"""Producer agent — Music Theory / Raga / Stem Engineer + Generative Sound
Designer orchestration node for the Autonomous Remix Studio.

Role in the LangGraph workflow
--------------------------------
``producer_node`` runs after the A&R Director has filled ``bpm`` / ``key`` /
``raga`` / ``tala`` / ``blueprint``. It produces the raw material the
Mixing Engineer later assembles:

* **Stems** — the isolated layers of the source song.
* **Generated stems** — new tempo/key-locked material (beats, risers) the
  Director's blueprint calls for.

Two paths
---------
1. **Primary path (MCP servers).** Stem separation and generation are
   delegated to the specialised MCP servers, when they exist and answer:

   * ``mcp_servers/stem_theory_server.py`` — ``separate_stems(audio_path,
     model="BS-RoFormer")`` returning ``{vocals, backing_vocals, drums,
     bass, melodic}`` paths.
   * ``mcp_servers/generative_audio_server.py`` — ``generate_beat(bpm, key,
     bars, style)`` / ``generate_riser(duration_ms, style)`` returning
     audio paths.

   Server locations are configurable through
   ``state["mix_params"]["mcp_servers"]`` (a dict with ``"stem_theory"``
   and ``"generative"`` keys, or a directory path); they default to the
   project's ``mcp_servers/`` directory. Every MCP call is wrapped so a
   missing server, a failed launch or a tool error degrades cleanly to the
   fallback — the graph never crashes because a model server is down.

2. **Fallback path (real DSP, no stubs).** When a server is unavailable:

   * Stems: librosa HPSS (harmonic / percussive separation) plus a
     spectral bass estimate (zero-phase low-pass of the harmonic stem
     below ~150 Hz), written as 24-bit WAVs.
   * Generation: numpy synthesis — a tempo-locked kick pattern (sine
     burst with pitch envelope, placed on a uniform grid anchored to the
     first strong low-band onset of the percussive/drum stem) and a
     filtered-noise riser (time-varying low-pass sweep + amplitude ramp).

   The fallback appends a clear note to ``state["notes"]`` saying the
   full 5-stem BS-RoFormer separation / MusicGen generation needs the
   corresponding MCP server. Diagnostic notes (server missing, fallback
   used, partial-generation gaps) live in ``notes``; ``errors`` carries
   only genuine failures (unreadable input, a fallback that itself
   failed).

Invariants
----------
* Sample rate is 44 100 Hz everywhere; all timestamps are converted to
  exact sample indices with ``core.dsp_utils.ms_to_samples``.
* All file I/O goes through ``soundfile``; outputs are 24-bit WAV.
* ``producer_node`` returns a partial state update
  ``{"stems": {...}, "generated_stems": {...}, "errors": [...],
  "notes": [...]}`` — the ``errors`` and ``notes`` lists are appended by
  the LangGraph ``operator.add`` reducer. ``errors`` is reserved for
  genuine failures; diagnostics go to ``notes`` (D6).
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Import shim: make `agents.state` / `core.dsp_utils` importable whether this
# file is run as a script or imported as a module from the project root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.state import RemixAgentState, SAMPLE_RATE  # noqa: E402
from core.dsp_utils import ms_to_samples  # noqa: E402

__all__ = ["producer_node"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default stem-server / generative-server filenames (primary path).
STEM_SERVER_FILENAME = "stem_theory_server.py"
GENERATIVE_SERVER_FILENAME = "generative_audio_server.py"

#: Tool names on the (not-yet-built) MCP servers — the documented contracts
#: this node calls against. Do not rename without updating the servers.
STEM_TOOL = "separate_stems"
BEAT_TOOL = "generate_beat"
RISER_TOOL = "generate_riser"

#: Fallback stem keys (HPSS + spectral bass estimate).
FALLBACK_STEM_NAMES = ("harmonic", "percussive", "bass_estimate")

#: Bass estimate cutoff (Hz) for the low-pass of the harmonic stem.
BASS_CUTOFF_HZ = 150.0

#: Default riser length when the blueprint gives no build-up to measure.
DEFAULT_RISER_MS = 4000

#: Default key when the Director has not filled one and no estimate exists.
DEFAULT_KEY = "A minor"

#: MCP call budgets (seconds). Stem separation loads a heavy model; give it
#: room, but never block the graph forever.
STEM_TIMEOUT_S = 900.0
GENERATE_TIMEOUT_S = 600.0

#: Low-band cutoff (Hz) for kick-onset detection. Kicks live below ~200 Hz;
#: hats and melody do not, so the detector sees the kick train only.
LOW_BAND_CUTOFF_HZ = 200.0

#: Onset-detection hop (samples): 256 @ 44.1 kHz ~= 5.8 ms resolution.
ONSET_HOP_LENGTH = 256

#: Note appended to ``errors`` when the stem server is unavailable.
STEM_FALLBACK_NOTE = (
    "stem server unavailable — used librosa HPSS fallback "
    "(harmonic/percussive/bass_estimate); full 5-stem BS-RoFormer "
    "separation needs mcp_servers/stem_theory_server.py"
)

#: Note appended to ``errors`` when the generative server is unavailable.
GENERATIVE_FALLBACK_NOTE = (
    "generative server unavailable — synthesized kick_pattern + riser with "
    "numpy DSP fallback; MusicGen / Stable Audio generation needs "
    "mcp_servers/generative_audio_server.py"
)


# ---------------------------------------------------------------------------
# Small file / audio helpers
# ---------------------------------------------------------------------------


def _read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a WAV as ``(samples, channels)`` float64 plus its sample rate.

    Raises ``FileNotFoundError`` / ``ValueError`` with a clear message so
    the caller can record the failure in ``errors`` instead of crashing.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"input audio not found: {path}")
    audio, sr = sf.read(str(path), always_2d=True, dtype="float64")
    if sr != SAMPLE_RATE:
        raise ValueError(
            f"expected {SAMPLE_RATE} Hz studio-wide, got {sr} Hz: {path}"
        )
    return audio, sr


def _write_wav(path: str | Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    """Write ``(samples, channels)`` float audio as 24-bit WAV; return str path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float64), sr, subtype="PCM_24")
    return str(path)


def _resolve_output_dir(state: RemixAgentState) -> Path:
    """Run output directory: explicit ``mix_params["output_dir"]`` or a
    ``<input-stem>_producer`` folder next to the input audio."""
    explicit = state.get("mix_params", {}).get("output_dir")
    if explicit:
        return Path(explicit)
    input_path = Path(state["input_audio_path"])
    return input_path.parent / f"{input_path.stem}_producer"


def _resolve_server_paths(state: RemixAgentState) -> dict[str, Path]:
    """Resolve the two MCP server script paths.

    ``state["mix_params"]["mcp_servers"]`` may be:

    * a dict with ``"stem_theory"`` / ``"generative"`` keys (aliases
      ``"stem"`` / ``"gen"`` accepted), each a script path or a directory
      containing the script;
    * a single directory / script path (used for both servers);
    * absent — then both default to the project's ``mcp_servers/`` dir.
    """
    default_dir = _PROJECT_ROOT / "mcp_servers"
    configured = state.get("mix_params", {}).get("mcp_servers")

    def _pick(entry: Any, filename: str) -> Path:
        if entry is None:
            return default_dir / filename
        p = Path(entry)
        return p / filename if p.is_dir() else p

    if isinstance(configured, dict):
        stem_entry = configured.get("stem_theory", configured.get("stem"))
        gen_entry = configured.get("generative", configured.get("gen"))
        return {
            "stem_theory": _pick(stem_entry, STEM_SERVER_FILENAME),
            "generative": _pick(gen_entry, GENERATIVE_SERVER_FILENAME),
        }
    if isinstance(configured, (str, Path)):
        return {
            "stem_theory": _pick(configured, STEM_SERVER_FILENAME),
            "generative": _pick(configured, GENERATIVE_SERVER_FILENAME),
        }
    return {
        "stem_theory": default_dir / STEM_SERVER_FILENAME,
        "generative": default_dir / GENERATIVE_SERVER_FILENAME,
    }


def _style_string(state: RemixAgentState) -> str:
    """Human-readable style for generation calls.

    Built from the blueprint's first Drop section genre fusion (or the
    global fusion rules), the creative direction, and — when the Director
    identified them — the raga / tala. Never invents a raga or tala.
    """
    parts: list[str] = []
    blueprint = state.get("blueprint")
    if blueprint is not None:
        drops = blueprint.drop_sections
        if drops and drops[0].genre_fusion:
            parts.append(" + ".join(drops[0].genre_fusion))
        elif blueprint.genre_fusion_rules:
            rule = blueprint.genre_fusion_rules[0]
            parts.append(" + ".join(rule.genres))
    if state.get("creative_direction"):
        parts.append(state["creative_direction"])
    if state.get("raga"):
        parts.append(f"raga {state['raga']}")
    if state.get("tala"):
        parts.append(f"tala {state['tala']}")
    return ", ".join(parts) if parts else "electronic"


def _bars_for_generation(state: RemixAgentState, n_samples: int) -> int:
    """Bars of generated material to request: the blueprint's total when
    available, else the bars the source audio spans at the track BPM."""
    blueprint = state.get("blueprint")
    if blueprint is not None:
        return max(1, blueprint.total_bars)
    bpm = state.get("bpm") or 120.0
    bar_ms = 4 * 60_000.0 / bpm
    duration_ms = n_samples / SAMPLE_RATE * 1000.0
    return max(1, int(round(duration_ms / bar_ms)))


def _riser_duration_ms(state: RemixAgentState) -> int:
    """Riser length: the build-up immediately before the first Drop, else
    the default. Clamped to at least 500 ms."""
    blueprint = state.get("blueprint")
    if blueprint is not None and blueprint.drop_sections:
        first_drop = blueprint.drop_sections[0]
        for section in blueprint.sections:
            if section.end_ms == first_drop.start_ms:
                return max(500, section.duration_ms)
    return DEFAULT_RISER_MS


# ---------------------------------------------------------------------------
# Async / MCP plumbing
# ---------------------------------------------------------------------------


def _run_async(coro: Any) -> Any:
    """Drive an async coroutine from sync node code.

    ``asyncio.run`` when there is no running loop (the normal LangGraph
    ``invoke`` case); otherwise run it on a fresh loop in a helper thread.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        result: dict[str, Any] = {}
        error: dict[str, BaseException] = {}

        def _runner() -> None:
            try:
                result["value"] = asyncio.run(coro)
            except BaseException as exc:  # noqa: BLE001 — re-raised below
                error["error"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()
        if error:
            raise error["error"]
        return result.get("value")


def _extract_tool_result(result: Any) -> Any:
    """Pull the payload out of a FastMCP ``CallToolResult``.

    Prefers the parsed ``.data``; falls back to JSON-parsing the first
    text content block, then to the raw string. Returns ``None`` when the
    result carries nothing usable.
    """
    data = getattr(result, "data", None)
    if data is not None:
        return data
    content = getattr(result, "content", None) or []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except (ValueError, TypeError):
                return text
    return None


def _normalize_path_result(
    payload: Any, keys: tuple[str, ...]
) -> str | None:
    """Normalize a tool result that should name one audio file.

    Accepts a bare path string or a dict carrying one of ``keys``
    (e.g. ``output_path`` / ``path``). Returns the path string, or
    ``None`` when nothing usable is present.
    """
    if isinstance(payload, str) and payload.strip():
        return payload.strip()
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _normalize_stem_result(payload: Any) -> dict[str, str] | None:
    """Normalize a ``separate_stems`` result to ``{stem_name: path}``.

    Accepts either a flat dict of stem -> path, or a dict wrapping it
    under a ``"stems"`` key. Every value must be a non-empty string.
    """
    if isinstance(payload, dict) and isinstance(payload.get("stems"), dict):
        payload = payload["stems"]
    if not isinstance(payload, dict) or not payload:
        return None
    stems = {
        str(name): str(path)
        for name, path in payload.items()
        if isinstance(path, str) and path.strip()
    }
    return stems or None


async def _call_mcp_tool(
    server_path: Path,
    tool_name: str,
    arguments: dict[str, Any],
    timeout_s: float,
) -> Any:
    """Call one tool on an MCP server script over stdio; return its payload.

    Raises ``FileNotFoundError`` when the server script does not exist and
    propagates any launch / tool error to the caller for fallback handling.
    """
    if not server_path.is_file():
        raise FileNotFoundError(f"MCP server script not found: {server_path}")
    # Imported lazily so the node still imports (and its fallback still
    # runs) on systems without FastMCP installed.
    from fastmcp import Client

    # Pass the Path (not a string): FastMCP infers stdio transport from a
    # Path and string inference is deprecated (removed in FastMCP 5).
    async with Client(server_path, timeout=timeout_s) as client:
        result = await client.call_tool(tool_name, arguments, timeout=timeout_s)
    return _extract_tool_result(result)


def _call_mcp_tool_sync(
    server_path: Path,
    tool_name: str,
    arguments: dict[str, Any],
    timeout_s: float,
) -> Any:
    """Sync wrapper around :func:`_call_mcp_tool` for the graph node."""
    return _run_async(_call_mcp_tool(server_path, tool_name, arguments, timeout_s))


# ---------------------------------------------------------------------------
# Primary path — MCP servers
# ---------------------------------------------------------------------------


def _try_primary_stems(
    state: RemixAgentState,
    server_path: Path,
) -> tuple[dict[str, str] | None, str | None]:
    """Separate stems via the stem-theory MCP server.

    Returns ``(stems, error_note)`` — on success ``error_note`` is None;
    on any failure the note explains why the fallback was used.
    """
    try:
        payload = _call_mcp_tool_sync(
            server_path,
            STEM_TOOL,
            {"audio_path": state["input_audio_path"], "model": "BS-RoFormer"},
            STEM_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001 — any MCP failure falls through
        return None, f"stem server call failed ({type(exc).__name__}: {exc})"
    stems = _normalize_stem_result(payload)
    if stems is None:
        return None, f"stem server returned no usable stems: {payload!r}"
    missing = [name for name, path in stems.items() if not Path(path).is_file()]
    if missing:
        return None, f"stem server returned missing files for stems: {missing}"
    return stems, None


def _try_primary_generation(
    state: RemixAgentState,
    server_path: Path,
    bpm: float,
    key: str,
    bars: int,
    riser_ms: int,
) -> tuple[dict[str, str] | None, str | None]:
    """Generate beat + riser via the generative MCP server.

    Returns ``(generated_stems, error_note)``. A partial success (beat
    OK, riser failed) still returns the beat and notes the riser failure —
    the caller fills the gap from the fallback instead of discarding it.
    """
    style = _style_string(state)
    generated: dict[str, str] = {}
    notes: list[str] = []
    try:
        payload = _call_mcp_tool_sync(
            server_path,
            BEAT_TOOL,
            {"bpm": bpm, "key": key, "bars": bars, "style": style},
            GENERATE_TIMEOUT_S,
        )
        beat_path = _normalize_path_result(
            payload, ("output_path", "path", "beat_path", "audio_path")
        )
        if beat_path and Path(beat_path).is_file():
            generated["beat"] = beat_path
        else:
            notes.append(f"generate_beat returned no usable file: {payload!r}")
    except Exception as exc:  # noqa: BLE001 — any MCP failure falls through
        notes.append(f"generate_beat failed ({type(exc).__name__}: {exc})")
    try:
        payload = _call_mcp_tool_sync(
            server_path,
            RISER_TOOL,
            {"duration_ms": riser_ms, "style": style},
            GENERATE_TIMEOUT_S,
        )
        riser_path = _normalize_path_result(
            payload, ("output_path", "path", "riser_path", "audio_path")
        )
        if riser_path and Path(riser_path).is_file():
            generated["riser"] = riser_path
        else:
            notes.append(f"generate_riser returned no usable file: {payload!r}")
    except Exception as exc:  # noqa: BLE001 — any MCP failure falls through
        notes.append(f"generate_riser failed ({type(exc).__name__}: {exc})")
    if not generated:
        return None, "; ".join(notes) if notes else "generative server failed"
    return generated, ("; ".join(notes) if notes else None)


# ---------------------------------------------------------------------------
# Fallback path — real DSP (librosa HPSS + numpy synthesis)
# ---------------------------------------------------------------------------


def _fallback_stems(
    audio: np.ndarray,
    sr: int,
    out_dir: Path,
) -> dict[str, str]:
    """HPSS stem separation + spectral bass estimate.

    ``audio`` is ``(samples, channels)`` float64. HPSS runs per channel
    (librosa works on 1-D signals); the bass estimate is a zero-phase
    4th-order Butterworth low-pass of the harmonic stem below
    ``BASS_CUTOFF_HZ``. Returns ``{harmonic, percussive, bass_estimate}``
    as 24-bit stereo WAV paths.
    """
    import librosa
    from scipy.signal import butter, sosfiltfilt

    stems: dict[str, str] = {}
    harmonic_ch: list[np.ndarray] = []
    percussive_ch: list[np.ndarray] = []
    for ch in range(audio.shape[1]):
        harmonic, percussive = librosa.effects.hpss(audio[:, ch])
        harmonic_ch.append(np.asarray(harmonic, dtype=np.float64))
        percussive_ch.append(np.asarray(percussive, dtype=np.float64))
    harmonic = np.column_stack(harmonic_ch)
    percussive = np.column_stack(percussive_ch)

    sos = butter(4, BASS_CUTOFF_HZ, btype="lowpass", fs=sr, output="sos")
    bass = np.column_stack(
        [sosfiltfilt(sos, harmonic[:, ch]) for ch in range(harmonic.shape[1])]
    )

    stems["harmonic"] = _write_wav(out_dir / "harmonic.wav", harmonic, sr)
    stems["percussive"] = _write_wav(out_dir / "percussive.wav", percussive, sr)
    stems["bass_estimate"] = _write_wav(out_dir / "bass_estimate.wav", bass, sr)
    return stems


def _refine_anchor(
    grid_mono: np.ndarray,
    sr: int,
    candidate_sample: int,
) -> int:
    """Refine a flux-detected onset to the true attack (sample index).

    The STFT-flux peak lands ~11-18 ms BEFORE the transient (the flux
    rises when the transient enters the 46 ms STFT window), so a grid
    anchored to the raw flux candidate would sit ~18 ms ahead of the
    source kicks — and the critic would see two interleaved trains. The
    true attack is found from the raw signal's 1 ms RMS envelope: the
    local max in the 25 ms after the candidate is the kick itself, and
    the first 50%-of-max crossing scanning forward is its attack to
    ~1 ms (measured: flux candidate 0.052 s -> refined 0.0708 s for a
    true first kick at 0.070 s). Hats stay below the 50% bar. Returns the
    flux candidate unchanged when no crossing is found.
    """
    win = max(1, int(round(0.001 * sr)))
    n_win = len(grid_mono) // win
    if n_win == 0:
        return candidate_sample
    env = np.sqrt(
        np.mean(grid_mono[: n_win * win].reshape(-1, win) ** 2, axis=1)
    )
    lo = candidate_sample // win
    hi = min(len(env), int((candidate_sample / sr + 0.025) * sr / win) + 1)
    if hi <= lo:
        return candidate_sample
    local_max = float(np.max(env[lo:hi]))
    if local_max <= 0.0:
        return candidate_sample
    threshold = 0.5 * local_max
    for i in range(lo, hi):
        if env[i] >= threshold:
            return int(i * win)
    return candidate_sample


def _low_band_flux(audio_mono: np.ndarray, sr: int) -> np.ndarray:
    """Positive spectral flux of the <= 200 Hz STFT bins, one value/frame.

    Mirrors ``critic_agent._low_band_flux`` deliberately, so the Producer
    and the Critic measure the same thing: a steady tone has constant
    magnitude in its bin and contributes ~zero flux however loud it is,
    while a kick transient jumps the low-bin magnitudes. (Low-passing in
    the time domain first would be wrong two ways: the zero-phase filter
    pre-rings ~25 ms before each kick, so detected onsets land early by
    a varying amount and fail any uniformity check; and two steady tones
    beating against each other modulate the low-passed envelope and look
    like onsets.)
    """
    import librosa

    spec = np.abs(librosa.stft(audio_mono, n_fft=2048, hop_length=ONSET_HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low = spec[freqs <= LOW_BAND_CUTOFF_HZ]
    diffs = np.diff(low, axis=1, prepend=low[:, :1])
    return np.sum(np.maximum(diffs, 0.0), axis=0)


def _low_band_onsets(
    audio_mono: np.ndarray,
    sr: int,
    bpm: float,
) -> list[int]:
    """Strong low-band onset positions (samples) in a mono signal.

    Same measurement as the Critic's beat-drift check: positive spectral
    flux of the <= 200 Hz STFT bins, peak-picked without backtrack, keeping
    only strong onsets (flux >= max(5x the median frame flux, 5% of the
    peak) — the same bar the Critic uses, so the Producer's anchor is the
    Critic's anchor). The absolute detected times run ~18 ms early (the
    flux peaks when the transient enters the 46 ms STFT window), but the
    bias is constant across kicks, so it cancels in every relative
    measurement — and the Critic anchors its grid at the first detected
    downbeat, not at sample 0. Onsets closer together than 0.6 of a beat
    are de-duplicated (first one wins): off-beat hats sit ~0.25 s after
    the kick at 120 BPM and must not read as extra downbeats. Returns an
    empty list when no downbeats are detectable (flat flux envelope).
    """
    import librosa

    if audio_mono.size == 0 or not np.any(audio_mono):
        return []
    try:
        env = _low_band_flux(np.asarray(audio_mono, dtype=np.float64), sr)
    except Exception:  # noqa: BLE001 — detection is best-effort
        return []
    if env.size == 0:
        return []
    env_median = float(np.median(env))
    env_max = float(np.max(env))
    if env_max <= 0.0 or env_max <= 10.0 * env_median:
        return []
    try:
        onsets = librosa.onset.onset_detect(
            onset_envelope=env, sr=sr, units="samples",
            hop_length=ONSET_HOP_LENGTH, backtrack=False,
        )
    except Exception:  # noqa: BLE001 — detection is best-effort
        return []
    detected = [int(o) for o in np.atleast_1d(onsets)]
    if not detected:
        return []
    frames = np.clip(
        librosa.time_to_frames(
            np.asarray(detected, dtype=float) / sr, sr=sr,
            hop_length=ONSET_HOP_LENGTH,
        ),
        0, len(env) - 1,
    )
    strength = env[frames]
    threshold = max(5.0 * env_median, 0.05 * env_max)
    strong = [o for o, s in zip(detected, strength) if s >= threshold]
    if not strong:
        return []
    min_gap = max(1, int(round(0.6 * sr * 60.0 / bpm)))
    kept: list[int] = []
    for onset in strong:
        if not kept or onset - kept[-1] >= min_gap:
            kept.append(onset)
    return kept


def _beat_samples(
    grid_mono: np.ndarray,
    sr: int,
    bpm: float,
    n_samples: int,
) -> list[int]:
    """Downbeat sample positions for the kick grid.

    Anchored to the source audio, on an exact uniform grid. The anchor is
    the first strong low-band onset in ``grid_mono`` — the percussive/drum
    stem when available, else the mix — refined to the true attack with
    :func:`_refine_anchor` (the STFT-flux peak runs ~11-18 ms ahead of the
    transient, and an unrefined anchor would layer the pattern a constant
    ~18 ms ahead of the source kicks). The source audio is ground truth:
    the old code trusted ``librosa.beat.beat_track``'s first-beat phase,
    which can land ~160 ms late and make the first interval 0.592 s at
    120 BPM, so the master layered two kick trains ~90-230 ms apart and
    the critic (correctly) rejected it. Subsequent kicks sit on a uniform
    grid with spacing exactly 60/BPM, so by construction every interval
    is exactly one beat — a 0.592 s first interval is impossible (asserted
    below). The grid extends backwards from the anchor to sample 0 so the
    pattern covers the whole track. The grid is ALWAYS uniform: no
    fallback to the raw detected onsets — frame quantization makes the
    detected times drift ~0.75 ms/beat against the true grid, which would
    misfire any uniformity check on a clean track, and a genuinely
    drifting source is rejected by the critic either way (its own kicks
    drift), so following its onsets could not save the run. With no
    detectable onsets at all, falls back to a uniform grid from sample 0.
    """
    beat_len = max(1, int(round(sr * 60.0 / bpm)))
    onsets = _low_band_onsets(grid_mono, sr, bpm)
    if not onsets:
        return list(range(0, n_samples, beat_len))
    # The flux peak runs ~11-18 ms ahead of the transient; refine the
    # anchor to the true attack so the pattern lands exactly on the
    # source kicks (in-phase layering) instead of ~18 ms ahead of them.
    anchor = _refine_anchor(grid_mono, sr, int(onsets[0]))
    grid: list[int] = []
    k = 0
    while anchor - k * beat_len >= 0:
        grid.append(anchor - k * beat_len)
        k += 1
    k = 1
    while anchor + k * beat_len < n_samples:
        grid.append(anchor + k * beat_len)
        k += 1
    grid.sort()
    # Invariant: every interval is exactly one beat (sample-exact).
    assert all(b - a == beat_len for a, b in zip(grid, grid[1:])), (
        f"kick grid is not uniform: {grid[:4]} (beat_len={beat_len})"
    )
    return grid


def _synthesize_kick_pattern(
    n_samples: int,
    sr: int,
    bpm: float,
    beat_samples: list[int],
) -> np.ndarray:
    """Tempo-locked kick pattern: a sine burst with a pitch envelope
    (150 -> 50 Hz over ~30 ms) and exponential decay (~45 ms), placed at
    every downbeat sample. Mono, peak-normalized to 0.9."""
    out = np.zeros(n_samples, dtype=np.float64)
    burst_len = ms_to_samples(150, sr)
    t = np.arange(burst_len, dtype=np.float64) / sr
    # Pitch envelope: 150 Hz decaying to 50 Hz.
    freq = 50.0 + 100.0 * np.exp(-t / 0.03)
    phase = 2.0 * np.pi * np.cumsum(freq) / sr
    envelope = np.exp(-t / 0.045)
    burst = np.sin(phase) * envelope
    for start in beat_samples:
        if start >= n_samples:
            continue
        end = min(n_samples, start + burst_len)
        out[start:end] += burst[: end - start]
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out * (0.9 / peak)
    return out


def _synthesize_riser(
    duration_ms: int,
    sr: int,
) -> np.ndarray:
    """Noise riser: white noise through a time-varying one-pole low-pass
    whose cutoff sweeps 300 Hz -> 9 kHz, under a rising amplitude ramp.
    Mono, peak-normalized to 0.8."""
    rng = np.random.default_rng(0)  # deterministic: same input, same riser
    n = ms_to_samples(duration_ms, sr)
    noise = rng.standard_normal(n)
    # Exponential cutoff sweep.
    fc = np.geomspace(300.0, 9000.0, n)
    alpha = 1.0 - np.exp(-2.0 * np.pi * fc / sr)
    out = np.empty(n, dtype=np.float64)
    y = 0.0
    for i in range(n):  # one-pole filter must run sequentially
        y += alpha[i] * (noise[i] - y)
        out[i] = y
    ramp = np.linspace(0.0, 1.0, n) ** 2
    out = out * ramp
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out * (0.8 / peak)
    return out


def _grid_source_audio(
    stems: dict[str, str],
    audio: np.ndarray,
) -> np.ndarray:
    """Mono audio to anchor the kick grid on.

    Prefers the percussive/drum stem when it exists — the source's own
    kicks are ground truth, so the synthesized pattern lands exactly on
    them instead of on a beat-tracker's guessed phase. Falls back to the
    mix mono when no drum stem is available or readable.
    """
    for key in ("percussive", "drums"):
        path = stems.get(key) if stems else None
        if not path:
            continue
        try:
            stem_audio, _ = _read_wav(path)
        except Exception:  # noqa: BLE001 — fall through to the mix
            continue
        if stem_audio.size == 0 or not np.any(stem_audio):
            continue
        return (
            stem_audio.mean(axis=1)
            if stem_audio.shape[1] > 1
            else stem_audio[:, 0]
        )
    return audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]


def _fallback_generation(
    audio: np.ndarray,
    sr: int,
    bpm: float,
    riser_ms: int,
    out_dir: Path,
    grid_audio: np.ndarray | None = None,
) -> dict[str, str]:
    """Synthesized tempo-locked material: kick pattern + riser, as
    24-bit WAVs (mono up-mixed to stereo for downstream consistency).

    The kick grid is anchored to ``grid_audio`` (the percussive/drum stem
    when available, else the mix mono) — see :func:`_beat_samples`. The
    pattern therefore lands on the source's own kicks instead of on a
    beat-tracked phase that can drift away from them.
    """
    mono = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    grid_source = grid_audio if grid_audio is not None else mono
    beats = _beat_samples(grid_source, sr, bpm, audio.shape[0])
    kick = _synthesize_kick_pattern(audio.shape[0], sr, bpm, beats)
    riser = _synthesize_riser(riser_ms, sr)
    kick_stereo = np.column_stack([kick, kick])
    riser_stereo = np.column_stack([riser, riser])
    return {
        "kick_pattern": _write_wav(out_dir / "kick_pattern.wav", kick_stereo, sr),
        "riser": _write_wav(out_dir / "riser.wav", riser_stereo, sr),
    }


# ---------------------------------------------------------------------------
# Tempo / key resolution
# ---------------------------------------------------------------------------


def _resolve_bpm_key(
    state: RemixAgentState,
    audio: np.ndarray,
    sr: int,
) -> tuple[float, str]:
    """BPM/key from state (Director-filled) when present; otherwise estimate
    tempo with librosa and default the key. Never raises — a missing tempo
    falls back to 120 BPM so the node can still produce material."""
    bpm = state.get("bpm")
    if bpm is None or bpm <= 0:
        try:
            import librosa

            mono = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
            estimated, _ = librosa.beat.beat_track(y=mono, sr=sr)
            bpm = float(np.atleast_1d(estimated)[0])
        except Exception:  # noqa: BLE001 — estimation is best-effort
            bpm = 120.0
        if not bpm or bpm <= 0:
            bpm = 120.0
    key = state.get("key") or DEFAULT_KEY
    return float(bpm), str(key)


# ---------------------------------------------------------------------------
# Graph node
# ---------------------------------------------------------------------------


def producer_node(state: RemixAgentState) -> dict:
    """LangGraph node: produce stems + generated material for the remix.

    Orchestration order: (1) read the source audio and resolve BPM/key,
    (2) try the stem-theory MCP server, falling back to HPSS DSP,
    (3) try the generative MCP server, falling back to numpy synthesis.
    Genuine failures (unreadable input, a fallback that itself failed)
    are recorded in ``errors``; diagnostics — server missing, fallback
    used, partial-generation gaps — go to ``notes`` (D6). The node always
    returns a partial state update and never raises into the graph.
    """
    errors: list[str] = []
    notes: list[str] = []
    try:
        audio, sr = _read_wav(state["input_audio_path"])
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        return {
            "stems": {},
            "generated_stems": {},
            "errors": [f"producer: cannot read input audio: {exc}"],
            "notes": [],
        }

    out_dir = _resolve_output_dir(state)
    out_dir.mkdir(parents=True, exist_ok=True)
    bpm, key = _resolve_bpm_key(state, audio, sr)
    servers = _resolve_server_paths(state)

    # --- Stems: primary (BS-RoFormer MCP) then HPSS fallback. ---
    stems, stem_note = _try_primary_stems(state, servers["stem_theory"])
    if stems is None:
        notes.append(f"producer: {stem_note or 'stem server unavailable'}")
        try:
            stems = _fallback_stems(audio, sr, out_dir)
            notes.append(f"producer: {STEM_FALLBACK_NOTE}")
        except Exception as exc:  # noqa: BLE001 — never crash the graph
            errors.append(f"producer: HPSS fallback failed: {exc}")
            stems = {}

    # --- Generation: primary (MusicGen/Stable Audio MCP) then synthesis. ---
    # Kick-grid anchor: the source's own percussive/drum stem when we have
    # one — its kicks are ground truth for the synthesized pattern.
    grid_audio = _grid_source_audio(stems, audio)
    bars = _bars_for_generation(state, audio.shape[0])
    riser_ms = _riser_duration_ms(state)
    generated, gen_note = _try_primary_generation(
        state, servers["generative"], bpm, key, bars, riser_ms
    )
    if generated is None:
        notes.append(f"producer: {gen_note or 'generative server unavailable'}")
        try:
            generated = _fallback_generation(
                audio, sr, bpm, riser_ms, out_dir, grid_audio=grid_audio
            )
            notes.append(f"producer: {GENERATIVE_FALLBACK_NOTE}")
        except Exception as exc:  # noqa: BLE001 — never crash the graph
            errors.append(f"producer: synthesis fallback failed: {exc}")
            generated = {}
    elif gen_note:
        # Partial primary success (e.g. beat OK, riser failed): fill the
        # missing piece from the synthesis fallback.
        try:
            fallback = _fallback_generation(
                audio, sr, bpm, riser_ms, out_dir, grid_audio=grid_audio
            )
            for name, path in fallback.items():
                generated.setdefault(name, path)
            notes.append(f"producer: {gen_note}")
            notes.append(
                "producer: partial generative result — missing pieces "
                "synthesized with numpy DSP fallback"
            )
        except Exception as exc:  # noqa: BLE001 — keep the partial result
            errors.append(f"producer: synthesis fill-in failed: {exc}")

    return {
        "stems": stems,
        "generated_stems": generated,
        "errors": errors,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Self-test (run: python -m agents.producer_agent)
# ---------------------------------------------------------------------------


def _build_test_blueprint() -> Any:
    """Minimal valid 30 s / 128 BPM / A minor blueprint for the self-test.

    16 bars tile exactly (bar = 1875 ms): Intro 4, Build-up 4, Drop 4,
    Breakdown 2, Outro 2. The 30 s hook spans the whole track; the 15 s
    hook covers the Drop + Breakdown.
    """
    from agents.state import (
        ArrangementBlueprint,
        GenreFusionRule,
        Section,
        SectionType,
        ViralHookWindow,
    )

    return ArrangementBlueprint(
        title="Producer Self-Test",
        track_duration_ms=30000,
        bpm=128.0,
        key="A minor",
        sections=[
            Section(
                name=SectionType.INTRO,
                start_ms=0,
                end_ms=7500,
                energy=0.2,
                bpm=128.0,
                key="A minor",
                bars=4,
                genre_fusion=["kuthu", "phonk"],
            ),
            Section(
                name=SectionType.BUILD_UP,
                start_ms=7500,
                end_ms=15000,
                energy=0.5,
                bpm=128.0,
                key="A minor",
                bars=4,
                genre_fusion=["kuthu", "phonk"],
            ),
            Section(
                name=SectionType.DROP,
                start_ms=15000,
                end_ms=22500,
                energy=1.0,
                bpm=128.0,
                key="A minor",
                bars=4,
                genre_fusion=["kuthu", "phonk"],
            ),
            Section(
                name=SectionType.BREAKDOWN,
                start_ms=22500,
                end_ms=26250,
                energy=0.4,
                bpm=128.0,
                key="A minor",
                bars=2,
                genre_fusion=["kuthu", "phonk"],
            ),
            Section(
                name=SectionType.OUTRO,
                start_ms=26250,
                end_ms=30000,
                energy=0.2,
                bpm=128.0,
                key="A minor",
                bars=2,
                genre_fusion=["kuthu", "phonk"],
            ),
        ],
        viral_hooks=[
            ViralHookWindow(
                duration_s=15,
                start_ms=15000,
                end_ms=30000,
                retention_score=0.9,
                reason="drop peak",
            ),
            ViralHookWindow(
                duration_s=30,
                start_ms=0,
                end_ms=30000,
                retention_score=0.8,
                reason="full track",
            ),
        ],
        genre_fusion_rules=[
            GenreFusionRule(
                genres=["kuthu", "phonk"],
                blend_ratio=0.6,
                applies_to=[SectionType.DROP],
            )
        ],
        raga="Mohanam",
        tala="Adi (8-beat)",
    )


def _make_test_wav(path: Path, duration_s: float = 15.0) -> str:
    """15 s synthetic source: 440 Hz melody + 220 Hz bass + periodic
    noise bursts (hat-like), stereo 44.1 kHz 24-bit WAV."""
    sr = SAMPLE_RATE
    n = ms_to_samples(int(duration_s * 1000), sr)
    t = np.arange(n, dtype=np.float64) / sr
    melody = 0.3 * np.sin(2.0 * np.pi * 440.0 * t) * (0.5 + 0.5 * np.sin(2.0 * np.pi * 2.0 * t))
    bass = 0.3 * np.sin(2.0 * np.pi * 110.0 * t)
    rng = np.random.default_rng(1)
    hats = np.zeros(n)
    hat_len = ms_to_samples(50, sr)
    for start in range(0, n, ms_to_samples(250, sr)):
        hats[start : start + hat_len] += rng.standard_normal(hat_len) * 0.2
    left = melody + bass + hats
    right = melody * 0.9 + bass + hats * 0.8
    stereo = np.column_stack([left, right])
    stereo /= np.max(np.abs(stereo))
    return _write_wav(path, stereo, sr)


def _self_test() -> None:
    """End-to-end test: 15 s synthetic WAV -> producer_node -> verify every
    returned path exists and is a valid WAV. Prints the result dicts."""
    from agents.state import create_initial_state

    tmp = Path("/tmp")
    test_wav = _make_test_wav(tmp / "producer_test_input.wav")
    blueprint = _build_test_blueprint()
    state = create_initial_state(test_wav, "kuthu phonk test remix")
    state["bpm"] = 128.0
    state["key"] = "A minor"
    state["raga"] = "Mohanam"
    state["tala"] = "Adi (8-beat)"
    state["blueprint"] = blueprint

    update = producer_node(state)
    print("stems:", json.dumps(update["stems"], indent=2))
    print("generated_stems:", json.dumps(update["generated_stems"], indent=2))
    print("errors:", json.dumps(update["errors"], indent=2))
    print("notes:", json.dumps(update["notes"], indent=2))

    # D6: the fallback path is a diagnostic, not a failure — errors must
    # be empty and the diagnostics must land in `notes`. Since the
    # Layer 3 MCP servers now exist, the self-test must pass EITHER way:
    # primary path (5 BS-RoFormer stems, no fallback notes) or fallback
    # path (HPSS + numpy DSP notes). A mixed outcome (primary stems,
    # DSP generation) also passes — what matters is that the node
    # produced real material and reported honestly how it got it.
    assert update["errors"] == [], f"errors not empty: {update['errors']}"
    assert update["notes"], "expected diagnostics in notes"
    has_fallback_notes = any("HPSS fallback" in n for n in update["notes"]) and any(
        "numpy DSP fallback" in n for n in update["notes"]
    )
    has_primary_stems = {"vocals", "drums", "bass", "melodic", "backing"} <= set(
        update["stems"]
    )
    assert has_fallback_notes or has_primary_stems, (
        "expected either the HPSS/numpy DSP fallback notes or the 5 "
        f"BS-RoFormer stems; notes={update['notes']!r} "
        f"stems={sorted(update['stems'])}"
    )

    all_paths = {**update["stems"], **update["generated_stems"]}
    assert all_paths, "producer_node returned no files"
    for name, path in all_paths.items():
        assert Path(path).is_file(), f"missing file for {name}: {path}"
        info = sf.info(path)
        assert info.samplerate == SAMPLE_RATE, f"{name}: sr {info.samplerate}"
        assert info.duration > 0, f"{name}: zero-length audio"
        print(f"OK {name}: {path} ({info.duration:.2f}s, {info.channels}ch, {info.subtype})")
    print("SELF-TEST PASSED")


if __name__ == "__main__":
    _self_test()
