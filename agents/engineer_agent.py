"""Surgical DSP Mixing & Mastering Engineer agent — Autonomous Remix Studio.

LangGraph node: :func:`engineer_node` takes the workflow state
(:class:`~agents.state.RemixAgentState`) after the Director (blueprint)
and Producer (stems + generated stems) have run, and renders the arranged
remix:

1. Assemble the mix section by section from the blueprint's millisecond
   timestamps (converted with :func:`core.dsp_utils.ms_to_samples`, so
   every edit lands on an exact sample index).
2. Per section: role-classified stems are panned
   (:func:`~core.dsp_utils.binaural_pan`), the instrumental bus is carved
   around the vocal stem (:func:`~core.dsp_utils.dynamic_vocal_carve`),
   and the non-kick bus is ducked on the kick during Drop sections
   (:func:`~core.dsp_utils.kick_triggered_sidechain`).
3. Sections are joined in blueprint order with a ~50 ms equal-power
   crossfade at each boundary; the result is exactly ``track_len`` samples.
4. The full track is mastered: a true-peak limiter stage (Spotify
   Pedalboard, only engaged when the predicted true peak after the
   loudness gain would breach the ceiling — peaky mixes otherwise force
   :func:`~core.dsp_utils.normalize_to_lufs` to scale the whole track
   down and miss the LUFS target), then
   :func:`~core.dsp_utils.normalize_to_lufs` (default -14 LUFS, -9 LUFS
   in club mode) under the true-peak ceiling, written as 24-bit WAV,
   and the blueprint's 30 s viral-hook window is cut from the same master.

The Critic agent can steer a re-render through ``mix_params``:
``limiter_ceiling_db`` (true-peak ceiling, default -1.0 dBTP — a stricter
value is enforced with an extra scale-down pass, since
``normalize_to_lufs`` itself caps at -1.0 dBTP),
``bass_gain_db`` (per-role gain on bass stems, default 0.0 dB) and
``beat_shift_ms`` (time-shift of the assembled mix before mastering,
default 0.0 ms, clamped to ±250 ms). All three default to a no-op, so a
run without them renders byte-identically to before. ``gain_trim_db``
is deliberately NOT a knob: normalization to the LUFS target would
erase a pre-normalization trim, and a post-normalization trim would
break the target — the Critic does not derive it.

DSP only — this agent loads no heavy models. Every DSP operation is
delegated to :mod:`core.dsp_utils`; nothing here reimplements it.

Missing stems are skipped and noted in ``errors`` — never fatal. If no
usable stems exist at all, no master is written and ``master_path`` is
``""`` with the reason in ``errors``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Import shim: make `agents.state` / `core.dsp_utils` importable whether this
# file is run as a script (`python agents/engineer_agent.py`, sys.path[0]
# is agents/) or as a module from the project root. The DSP functions
# themselves are NOT redefined here — they live in core/dsp_utils.py.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.state import (  # noqa: E402
    SAMPLE_RATE,
    ArrangementBlueprint,
    RemixAgentState,
    SectionType,
)
from core.dsp_utils import (  # noqa: E402
    binaural_pan,
    detect_clipping,
    dynamic_vocal_carve,
    kick_triggered_sidechain,
    measure_loudness,
    ms_to_samples,
    normalize_to_lufs,
    time_stretch_formant_preserving,
)

try:  # Pedalboard is a pinned project dependency; the limiter stage is
    # skipped (with a note in mix_params) if it is ever unavailable.
    from pedalboard import Limiter, Pedalboard

    _PEDALBOARD_AVAILABLE = True
except Exception:  # noqa: BLE001 - any import failure means "no limiter"
    _PEDALBOARD_AVAILABLE = False

__all__ = ["engineer_node"]

# ---------------------------------------------------------------------------
# Defaults (overridable per run through state["mix_params"])
# ---------------------------------------------------------------------------

#: Default streaming loudness target in LUFS.
DEFAULT_TARGET_LUFS: float = -14.0

#: Club-master loudness target in LUFS (``mix_params["club_master"]``).
CLUB_TARGET_LUFS: float = -9.0

#: Hard true-peak ceiling applied by the master (dBTP). Never exceeded.
#: Also the default for ``mix_params["limiter_ceiling_db"]``.
MASTER_CEILING_DBTP: float = -1.0

#: Default per-role bass gain in dB (``mix_params["bass_gain_db"]``).
#: 0.0 dB multiplies by exactly 1.0 — an exact no-op. The Critic sets
#: -3.0 dB on a clipping reject to trim the low end.
DEFAULT_BASS_GAIN_DB: float = 0.0

#: Default time-shift of the assembled mix in ms
#: (``mix_params["beat_shift_ms"]``). Positive delays, negative advances.
DEFAULT_BEAT_SHIFT_MS: float = 0.0

#: Beat-shift magnitude is clamped to this (the Critic clamps to ±250 ms
#: too; this is the defensive bound on the engineer side).
MAX_BEAT_SHIFT_MS: float = 250.0

#: Crossfade between sections, in milliseconds (equal-power, sample-accurate).
DEFAULT_CROSSFADE_MS: int = 50

#: Default sidechain ducking depth in dB during Drop sections.
DEFAULT_SIDECHAIN_DEPTH_DB: float = 6.0

#: Default dynamic vocal-carve depth in dB.
DEFAULT_CARVE_REDUCTION_DB: float = 3.0

#: Length mismatches up to this many samples (~1 ms) are trimmed/padded;
#: larger ones are fixed with formant-preserving time-stretch.
LENGTH_TOLERANCE_SAMPLES: int = 44

#: Role base gains before the per-section energy scaling.
ROLE_BASE_GAINS: dict[str, float] = {
    "vocal": 1.0,
    "kick": 1.0,
    "bass": 0.9,
    "drums": 0.85,
    "melodic": 0.75,
    "fx": 0.7,
}

#: Binaural placement per role, in degrees (-90 hard left .. +90 hard right).
ROLE_PAN_DEG: dict[str, float] = {
    "vocal": 0.0,
    "kick": 0.0,
    "bass": 0.0,
    "drums": 0.0,
    "melodic": 18.0,
    "fx": 0.0,
}

#: Substring rules for classifying a stem key into a mix role.
#: Order matters: kick is checked before drums, vocal before melodic.
_ROLE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("kick", ("kick",)),
    ("vocal", ("vocal", "vox", "voice", "choir", "harmony", "backing")),
    ("bass", ("bass", "sub", "808")),
    ("drums", ("drum", "perc", "snare", "hihat", "hat", "cymbal", "clap", "tom")),
    (
        "melodic",
        (
            "melod",
            "synth",
            "keys",
            "piano",
            "guitar",
            "pad",
            "strings",
            "brass",
            "flute",
            "instrument",
            "other",
            "riser",
            "fx",
            "lead",
        ),
    ),
]


#: Limiter thresholds (dBFS) tried in order when peak control is needed.
#: Each pass re-limits the already-limited mix, shaving peaks further.
LIMITER_THRESHOLDS_DB: tuple[float, ...] = (-1.0, -3.0, -6.0, -9.0)

#: Limiter release time in milliseconds.
LIMITER_RELEASE_MS: float = 100.0

#: A master counts as on-target when its measured LUFS is within this
#: many dB of the target (the test tolerance is ±0.6; the loop aims tighter).
MASTER_TOLERANCE_DB: float = 0.5


# ---------------------------------------------------------------------------
# Stem handling
# ---------------------------------------------------------------------------


def _classify_stem(key: str) -> str:
    """Map a stem dict key to a mix role (``vocal``/``drums``/``bass``/``melodic``/``kick``/``fx``).

    Unknown keys fall through to ``melodic`` — they still reach the mix,
    at the melodic gain, instead of being silently dropped.
    """
    lowered = key.lower()
    for role, needles in _ROLE_RULES:
        if any(needle in lowered for needle in needles):
            return role
    return "melodic"


def _to_core_layout(audio: np.ndarray) -> np.ndarray:
    """Convert soundfile layout ``(samples, channels)`` / ``(samples,)`` to
    core layout ``(channels, samples)`` as float64. Mono becomes stereo
    (duplicated channels) so every bus in the mix is stereo.
    """
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 1:
        return np.stack([a, a], axis=0)
    if a.ndim == 2:
        return np.stack([a[:, 0], a[:, 1]], axis=0) if a.shape[1] == 2 else np.stack(
            [a[:, 0], a[:, 0]], axis=0
        )
    raise ValueError(f"expected mono or stereo audio, got shape {a.shape}")


def _fit_exact_length(audio_2d: np.ndarray, target_len: int) -> np.ndarray:
    """Trim or zero-pad a ``(channels, samples)`` array to exactly
    ``target_len`` samples."""
    n = audio_2d.shape[1]
    if n == target_len:
        return audio_2d
    if n > target_len:
        return audio_2d[:, :target_len]
    out = np.zeros((audio_2d.shape[0], target_len), dtype=np.float64)
    out[:, :n] = audio_2d
    return out


def _load_stem(
    path: str, track_len: int, sr: int
) -> tuple[np.ndarray | None, str | None]:
    """Load one stem WAV and fit it to the track length.

    Returns ``(audio_2d, None)`` on success or ``(None, error_message)``.
    A stem whose sample rate is not 44100 Hz is skipped (the studio never
    resamples mid-pipeline). A length mismatch larger than
    ``LENGTH_TOLERANCE_SAMPLES`` is fixed with formant-preserving
    time-stretch; smaller ones are trimmed / zero-padded.
    """
    try:
        data, file_sr = sf.read(path, dtype="float64")
    except Exception as exc:  # unreadable / missing file, bad header, ...
        return None, f"could not read stem {path!r}: {exc}"
    if file_sr != sr:
        return None, (
            f"stem {path!r} is {file_sr} Hz, not {sr} Hz — skipped "
            "(studio never resamples mid-pipeline)"
        )
    audio = _to_core_layout(data)
    if audio.shape[1] == 0:
        return None, f"stem {path!r} is empty"
    length_diff = audio.shape[1] - track_len
    if abs(length_diff) > LENGTH_TOLERANCE_SAMPLES:
        rate = audio.shape[1] / track_len
        audio = time_stretch_formant_preserving(audio, sr, rate)
    return _fit_exact_length(audio, track_len), None


def _detect_kick_onsets(
    kick_audio_2d: np.ndarray, sr: int, *, min_gap_ms: float = 120.0
) -> list[int]:
    """Find kick hit positions (sample indices) in a kick stem.

    Mono-sums the stem, smooths the rectified envelope over ~5 ms, and
    takes rising edges above 25% of the peak envelope, enforcing a minimum
    gap so one hit is never counted twice.
    """
    mono = np.mean(kick_audio_2d, axis=0)
    env = np.abs(mono)
    win = max(1, int(round(0.005 * sr)))
    kernel = np.ones(win, dtype=np.float64) / win
    smooth = np.convolve(env, kernel, mode="same")
    peak = float(np.max(smooth)) if smooth.size else 0.0
    if peak <= 0.0:
        return []
    threshold = 0.25 * peak
    min_gap = int(round(min_gap_ms * sr / 1000.0))
    above = smooth > threshold
    onsets: list[int] = []
    last = -min_gap
    for i in range(1, len(above)):
        if above[i] and not above[i - 1] and i - last >= min_gap:
            onsets.append(i)
            last = i
    return onsets


def _grid_kick_triggers(blueprint: ArrangementBlueprint, sr: int) -> list[int]:
    """Fallback kick triggers: one kick per beat inside every Drop section,
    derived from each section's own BPM. Used when no kick stem exists.
    """
    triggers: list[int] = []
    for section in blueprint.drop_sections:
        beat_len = ms_to_samples(int(round(60_000.0 / section.bpm)), sr)
        if beat_len <= 0:
            continue
        start = ms_to_samples(section.start_ms, sr)
        end = ms_to_samples(section.end_ms, sr)
        triggers.extend(range(start, end, beat_len))
    return sorted(triggers)


# ---------------------------------------------------------------------------
# Mix-level corrective shift (Critic knob)
# ---------------------------------------------------------------------------


def _shift_mix(audio_2d: np.ndarray, sr: int, shift_ms: float) -> np.ndarray:
    """Shift a ``(2, n)`` mix by ``shift_ms`` before mastering.

    Positive values delay the mix (zero-fill at the start), negative
    values advance it (zero-fill at the end); the length stays exactly
    ``n`` samples. 0.0 ms returns the input unchanged. The Critic agent
    sets this to ``-median_offset_ms`` (clamped to ±250 ms) to re-anchor
    a mix whose onsets sit off the beat grid.
    """
    shift = int(round(float(shift_ms) * sr / 1000.0))
    if shift == 0:
        return audio_2d
    n = audio_2d.shape[1]
    out = np.zeros_like(audio_2d)
    if shift > 0:
        out[:, shift:] = audio_2d[:, : n - shift]
    else:
        out[:, : n + shift] = audio_2d[:, -shift:]
    return out


# ---------------------------------------------------------------------------
# Section rendering
# ---------------------------------------------------------------------------


def _render_section(
    stems: dict[str, np.ndarray],
    roles: dict[str, str],
    section_start: int,
    section_end: int,
    crossfade_tail: int,
    *,
    energy: float,
    is_drop: bool,
    kick_triggers: list[int],
    sidechain_depth_db: float,
    carve_reduction_db: float,
    section_gains: dict[str, float],
    section_name: str,
    bass_gain_db: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render one section (plus a ``crossfade_tail``-sample tail extension)
    from the full-track stems.

    Returns the rendered stereo ``(2, n)`` audio and a record of what was
    applied (gains, sidechain/carve flags, stems used).
    """
    sr = SAMPLE_RATE
    render_start = section_start
    render_end = section_end + crossfade_tail
    n = render_end - render_start

    # --- Split stems into buses -------------------------------------------
    vocal_bus = np.zeros((2, n), dtype=np.float64)
    inst_bus = np.zeros((2, n), dtype=np.float64)
    kick_bus = np.zeros((2, n), dtype=np.float64)
    used: list[str] = []
    for key, audio in stems.items():
        role = roles[key]
        seg = audio[:, render_start:render_end]
        gain = ROLE_BASE_GAINS[role] * (0.55 + 0.45 * energy)
        gain *= float(section_gains.get(section_name, 1.0))
        if role == "bass":
            # Critic corrective knob: trim the low end on a clipping
            # reject. 0.0 dB multiplies by exactly 1.0 — a no-op.
            gain *= 10.0 ** (bass_gain_db / 20.0)
        panned = binaural_pan(seg, sr, ROLE_PAN_DEG[role])
        seg = panned * gain
        used.append(key)
        if role == "vocal":
            vocal_bus += seg
        elif role == "kick":
            kick_bus += seg
        else:
            inst_bus += seg

    # --- Dynamic vocal carving: carve the vocal's frequencies out of the
    # instrumental bus (kick is never carved — it keeps its punch). --------
    carve_applied = bool(np.any(vocal_bus) and np.any(inst_bus))
    if carve_applied:
        inst_bus = dynamic_vocal_carve(
            inst_bus, vocal_bus, sr, bands=8, reduction_db=carve_reduction_db
        )

    # --- Sidechain: duck everything except the kick on every kick hit. ---
    non_kick = inst_bus + vocal_bus
    sidechain_applied = bool(is_drop and kick_triggers and np.any(non_kick))
    if sidechain_applied:
        local_triggers = [
            t - render_start for t in kick_triggers if render_start <= t < render_end
        ]
        if local_triggers:
            non_kick = kick_triggered_sidechain(
                non_kick,
                local_triggers,
                sr,
                depth_db=sidechain_depth_db,
            )
        else:
            sidechain_applied = False

    section_mix = non_kick + kick_bus
    record = {
        "name": section_name,
        "start_ms": None,  # filled in by the caller (has the blueprint ms)
        "gain": round(0.55 + 0.45 * energy, 3),
        "bass_gain_db": bass_gain_db,
        "sidechain_applied": sidechain_applied,
        "carve_applied": carve_applied,
        "stems_used": used,
    }
    return section_mix, record


# ---------------------------------------------------------------------------
# Mastering: true-peak limiter + LUFS normalize
# ---------------------------------------------------------------------------


def _apply_limiter(
    audio_2d: np.ndarray, sr: int, threshold_db: float
) -> np.ndarray:
    """Run a lookahead true-peak limiter (Pedalboard) over ``(2, n)`` audio.

    Peaks above ``threshold_db`` dBFS are shaved; everything below the
    threshold passes untouched, so dynamics are preserved whenever the
    limiter is not needed. Raises ``RuntimeError`` if Pedalboard is absent.
    """
    if not _PEDALBOARD_AVAILABLE:
        raise RuntimeError("pedalboard is not installed; cannot limit")
    board = Pedalboard(
        [Limiter(threshold_db=threshold_db, release_ms=LIMITER_RELEASE_MS)]
    )
    processed = board(audio_2d.T.astype(np.float32), sample_rate=sr)
    return np.asarray(processed, dtype=np.float64).T


def _enforce_ceiling(
    audio_2d: np.ndarray, sr: int, ceiling_dbtp: float
) -> np.ndarray:
    """Scale ``audio_2d`` down so its true peak is at most
    ``ceiling_dbtp - 0.05`` dBTP (the 0.05 dB margin keeps the ceiling
    honest against floating-point re-measurement). No-op when the peak
    already fits. Only used when the requested ceiling is stricter than
    -1.0 dBTP — :func:`~core.dsp_utils.normalize_to_lufs` already caps
    at -1.0 dBTP, so this pass exists solely for stricter ceilings.
    """
    tp = float(measure_loudness(audio_2d, sr)["true_peak_dbtp"])
    if tp <= ceiling_dbtp - 0.05:
        return audio_2d
    return audio_2d * 10.0 ** ((ceiling_dbtp - 0.05 - tp) / 20.0)


def _master_with_limiter(
    mix_2d: np.ndarray,
    sr: int,
    target_lufs: float,
    ceiling_dbtp: float = MASTER_CEILING_DBTP,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Master ``mix_2d`` to ``target_lufs`` under ``ceiling_dbtp`` dBTP.

    ``normalize_to_lufs`` alone cannot satisfy both constraints on a peaky
    mix: after gaining to the LUFS target, a hot true peak forces its
    ceiling pass to scale the whole track down, and the master lands
    several dB below target. So first predict the post-gain true peak; if
    it would breach the ceiling, shave the peaks with the limiter (trying
    progressively lower thresholds) and re-normalize until the master is
    within ``MASTER_TOLERANCE_DB`` of the target with true peak <= the
    ceiling. Digital silence passes through unchanged.

    ``ceiling_dbtp`` is the Critic's corrective knob
    (``mix_params["limiter_ceiling_db"]``, default -1.0 dBTP). A value
    looser than -1.0 dBTP is capped at -1.0 because
    ``normalize_to_lufs`` itself enforces -1.0 dBTP; a stricter value is
    enforced by :func:`_enforce_ceiling` after every normalize step.
    With the default ceiling every code path is exactly the old one, so
    default renders are byte-identical.

    Returns ``(mastered_audio, info)`` where ``info`` records whether the
    limiter ran, the threshold used, the ceiling applied, and the final
    measured loudness.
    """
    import math

    effective_ceiling = min(float(ceiling_dbtp), MASTER_CEILING_DBTP)
    strict = effective_ceiling < MASTER_CEILING_DBTP - 1e-9

    pre = measure_loudness(mix_2d, sr)
    pre_lufs = float(pre["integrated_lufs"])
    if not math.isfinite(pre_lufs):
        return mix_2d, {
            "limiter_applied": False,
            "limiter_backend": "none (digital silence)",
            "limiter_threshold_db": None,
            "limiter_iterations": 0,
            "limiter_ceiling_db": float(ceiling_dbtp),
            "effective_ceiling_dbtp": effective_ceiling,
        }
    gain_needed = target_lufs - pre_lufs
    predicted_tp = float(pre["true_peak_dbtp"]) + gain_needed
    if predicted_tp <= effective_ceiling - 0.05:
        mastered = normalize_to_lufs(mix_2d, sr, target_lufs)
        if strict:
            mastered = _enforce_ceiling(mastered, sr, effective_ceiling)
        measured = measure_loudness(mastered, sr)
        return mastered, {
            "limiter_applied": False,
            "limiter_backend": "not needed (predicted true peak "
            f"{predicted_tp:.2f} dBTP fits the ceiling)",
            "limiter_threshold_db": None,
            "limiter_iterations": 0,
            "limiter_ceiling_db": float(ceiling_dbtp),
            "effective_ceiling_dbtp": effective_ceiling,
            "predicted_true_peak_dbtp": round(predicted_tp, 2),
            "final_measured_lufs": round(float(measured["integrated_lufs"]), 2),
            "final_measured_true_peak_dbtp": round(
                float(measured["true_peak_dbtp"]), 2
            ),
        }

    if not _PEDALBOARD_AVAILABLE:
        mastered = normalize_to_lufs(mix_2d, sr, target_lufs)
        if strict:
            mastered = _enforce_ceiling(mastered, sr, effective_ceiling)
        measured = measure_loudness(mastered, sr)
        return mastered, {
            "limiter_applied": False,
            "limiter_backend": "none (pedalboard not installed)",
            "limiter_threshold_db": None,
            "limiter_iterations": 0,
            "limiter_ceiling_db": float(ceiling_dbtp),
            "effective_ceiling_dbtp": effective_ceiling,
            "predicted_true_peak_dbtp": round(predicted_tp, 2),
            "final_measured_lufs": round(float(measured["integrated_lufs"]), 2),
            "final_measured_true_peak_dbtp": round(
                float(measured["true_peak_dbtp"]), 2
            ),
        }

    current = mix_2d
    best: tuple[np.ndarray, dict[str, float]] | None = None
    for iteration, threshold in enumerate(LIMITER_THRESHOLDS_DB, start=1):
        current = _apply_limiter(current, sr, threshold)
        mastered = normalize_to_lufs(current, sr, target_lufs)
        if strict:
            mastered = _enforce_ceiling(mastered, sr, effective_ceiling)
        measured = measure_loudness(mastered, sr)
        info = {
            "limiter_applied": True,
            "limiter_backend": "pedalboard",
            "limiter_threshold_db": threshold,
            "limiter_iterations": iteration,
            "limiter_ceiling_db": float(ceiling_dbtp),
            "effective_ceiling_dbtp": effective_ceiling,
            "predicted_true_peak_dbtp": round(predicted_tp, 2),
            "final_measured_lufs": round(float(measured["integrated_lufs"]), 2),
            "final_measured_true_peak_dbtp": round(
                float(measured["true_peak_dbtp"]), 2
            ),
        }
        if (
            float(measured["integrated_lufs"]) >= target_lufs - MASTER_TOLERANCE_DB
            and float(measured["true_peak_dbtp"]) <= effective_ceiling
        ):
            return mastered, info
        if best is None or abs(
            float(measured["integrated_lufs"]) - target_lufs
        ) < abs(best[1]["final_measured_lufs"] - target_lufs):
            best = (mastered, info)
    assert best is not None  # ladder is non-empty by construction
    return best


# ---------------------------------------------------------------------------
# The LangGraph node
# ---------------------------------------------------------------------------


def engineer_node(state: RemixAgentState) -> dict:
    """Mix + master the arranged remix described by ``state``.

    Returns a partial state update::

        {"master_path": str, "mix_params": dict, "hook_path": str}

    plus ``"errors"`` (appended to the state's error channel) whenever
    something was skipped or failed. ``master_path`` / ``hook_path`` are
    ``""`` when no master could be rendered.

    ``mix_params`` in the returned update is the run's full audit trail:
    target LUFS, measured LUFS / true peak, sidechain depth, per-section
    gains and processing flags, stem roles, and output paths — everything
    the Critic agent needs to inspect the render.

    The Critic steers a re-render through ``mix_params`` knobs, all
    defaulting to a no-op: ``limiter_ceiling_db`` (true-peak ceiling,
    default -1.0 dBTP), ``bass_gain_db`` (per-role bass gain, default
    0.0 dB) and ``beat_shift_ms`` (time-shift of the assembled mix,
    default 0.0 ms, clamped to ±250 ms). ``gain_trim_db`` is NOT read —
    it is a dead knob by design (see the module docstring).
    """
    errors: list[str] = []
    incoming_params: dict[str, Any] = dict(state.get("mix_params") or {})
    sr = SAMPLE_RATE

    blueprint: ArrangementBlueprint | None = state.get("blueprint")
    if blueprint is None:
        errors.append("engineer: no ArrangementBlueprint in state; cannot arrange")
        return {
            "master_path": "",
            "mix_params": incoming_params,
            "hook_path": "",
            "errors": errors,
        }

    # --- Collect stems (separated + generated) ----------------------------
    all_stem_paths: dict[str, str] = {}
    all_stem_paths.update(state.get("stems") or {})
    all_stem_paths.update(state.get("generated_stems") or {})
    if not all_stem_paths:
        errors.append(
            "engineer: no stems in state['stems'] or state['generated_stems']; "
            "nothing to mix"
        )
        return {
            "master_path": "",
            "mix_params": incoming_params,
            "hook_path": "",
            "errors": errors,
        }

    track_len = ms_to_samples(blueprint.track_duration_ms, sr)
    stems: dict[str, np.ndarray] = {}
    roles: dict[str, str] = {}
    missing: list[str] = []
    for key, path in all_stem_paths.items():
        audio, err = _load_stem(str(path), track_len, sr)
        if err is not None:
            errors.append(f"engineer: skipped stem {key!r}: {err}")
            missing.append(key)
            continue
        assert audio is not None  # _load_stem returns audio iff err is None
        stems[key] = audio
        roles[key] = _classify_stem(key)
    if not stems:
        errors.append("engineer: every stem failed to load; nothing to mix")
        return {
            "master_path": "",
            "mix_params": incoming_params,
            "hook_path": "",
            "errors": errors,
        }

    # --- Run setup ---------------------------------------------------------
    target_lufs = incoming_params.get("target_lufs")
    club_master = bool(incoming_params.get("club_master", False))
    if target_lufs is None:
        target_lufs = CLUB_TARGET_LUFS if club_master else DEFAULT_TARGET_LUFS
    target_lufs = float(target_lufs)
    sidechain_depth_db = float(
        incoming_params.get("sidechain_depth_db", DEFAULT_SIDECHAIN_DEPTH_DB)
    )
    carve_reduction_db = float(
        incoming_params.get("carve_reduction_db", DEFAULT_CARVE_REDUCTION_DB)
    )
    crossfade_ms = int(incoming_params.get("crossfade_ms", DEFAULT_CROSSFADE_MS))
    crossfade = ms_to_samples(crossfade_ms, sr)
    section_gains: dict[str, float] = dict(
        incoming_params.get("section_gains") or {}
    )

    # --- Critic corrective knobs (all default to a no-op) -----------------
    limiter_ceiling_db = float(
        incoming_params.get("limiter_ceiling_db", MASTER_CEILING_DBTP)
    )
    bass_gain_db = float(
        incoming_params.get("bass_gain_db", DEFAULT_BASS_GAIN_DB)
    )
    beat_shift_ms = float(
        incoming_params.get("beat_shift_ms", DEFAULT_BEAT_SHIFT_MS)
    )
    beat_shift_ms = max(
        -MAX_BEAT_SHIFT_MS, min(MAX_BEAT_SHIFT_MS, beat_shift_ms)
    )

    run_id = str(
        incoming_params.get("run_id")
        or (f"{Path(str(state.get('input_audio_path') or 'track')).stem}-remix")
        or f"run-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    output_dir = Path(
        incoming_params.get("output_dir") or Path("outputs") / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Kick triggers: detect from a kick stem, else fall back to the
    # blueprint's beat grid inside Drop sections. ---------------------------
    kick_keys = [k for k, r in roles.items() if r == "kick"]
    kick_triggers: list[int] = []
    if kick_keys:
        kick_triggers = _detect_kick_onsets(stems[kick_keys[0]], sr)
        if not kick_triggers:
            errors.append(
                f"engineer: no onsets found in kick stem {kick_keys[0]!r}; "
                "falling back to the blueprint beat grid"
            )
    if not kick_triggers:
        kick_triggers = _grid_kick_triggers(blueprint, sr)

    # --- Render sections in blueprint order, crossfading ~50 ms at each
    # boundary (equal-power, sample-accurate; total length stays exactly
    # track_len because every section is placed at its absolute slot). -----
    sections = blueprint.sections
    full_mix = np.zeros((2, track_len), dtype=np.float64)
    section_records: list[dict[str, Any]] = []
    for idx, section in enumerate(sections):
        start = ms_to_samples(section.start_ms, sr)
        end = ms_to_samples(section.end_ms, sr)
        is_last = idx == len(sections) - 1
        tail = 0 if is_last else crossfade
        rendered, record = _render_section(
            stems,
            roles,
            start,
            end,
            tail,
            energy=float(section.energy),
            is_drop=section.name == SectionType.DROP,
            kick_triggers=kick_triggers,
            sidechain_depth_db=sidechain_depth_db,
            carve_reduction_db=carve_reduction_db,
            section_gains=section_gains,
            section_name=section.name.value,
            bass_gain_db=bass_gain_db,
        )
        record["start_ms"] = section.start_ms
        record["end_ms"] = section.end_ms
        section_records.append(record)

        head = rendered[:, : end - start]
        if idx == 0:
            full_mix[:, start:end] = head
        else:
            cf = min(crossfade, head.shape[1])
            if cf > 0:
                t = (np.arange(cf, dtype=np.float64) + 0.5) / cf
                fade_out = np.cos(t * np.pi / 2.0) ** 2
                fade_in = np.sin(t * np.pi / 2.0) ** 2
                full_mix[:, start : start + cf] = (
                    full_mix[:, start : start + cf] * fade_out
                    + head[:, :cf] * fade_in
                )
                full_mix[:, start + cf : end] = head[:, cf:]
            else:
                full_mix[:, start:end] = head
        if not is_last:
            # Tail extension becomes the crossfade material for the next
            # section's head; it is overwritten by the blend above.
            full_mix[:, end : end + tail] = rendered[:, end - start :]

    # --- Critic corrective knob: re-anchor the mix on the beat grid
    # (positive = delay, negative = advance; length stays track_len). ---
    if beat_shift_ms != 0.0:
        full_mix = _shift_mix(full_mix, sr, beat_shift_ms)

    # --- Master: limiter (only if the predicted true peak would breach
    # the ceiling) + loudness-normalize under the true-peak ceiling
    # (mix_params["limiter_ceiling_db"], default -1.0 dBTP). -------------
    pre_clip = detect_clipping(full_mix)
    mastered, limiter_info = _master_with_limiter(
        full_mix, sr, target_lufs, ceiling_dbtp=limiter_ceiling_db
    )
    measured = measure_loudness(mastered, sr)
    lufs_shortfall = target_lufs - float(measured["integrated_lufs"])
    if lufs_shortfall > MASTER_TOLERANCE_DB + 0.1:
        errors.append(
            "engineer: master landed "
            f"{lufs_shortfall:.2f} dB below the {target_lufs} LUFS target "
            f"(measured {float(measured['integrated_lufs']):.2f} LUFS); "
            "the true-peak ceiling forced the level down"
        )

    master_path = output_dir / "final_master_full.wav"
    sf.write(
        str(master_path), mastered.T, sr, subtype="PCM_24",
    )

    # --- Verify what was actually written (re-read from disk). ------------
    written_info = sf.info(str(master_path))
    written_data, written_sr = sf.read(str(master_path), dtype="float64")
    written_metrics = measure_loudness(_to_core_layout(written_data), sr)

    # --- Viral hook: cut the blueprint's 30 s window from the master. -----
    hook_30 = next(
        (h for h in blueprint.viral_hooks if h.duration_s == 30), None
    )
    hook_path = output_dir / "viral_hook_30s.wav"
    if hook_30 is not None:
        hook_start = ms_to_samples(hook_30.start_ms, sr)
        hook_end = min(hook_start + ms_to_samples(30_000, sr), track_len)
    else:
        hook_start, hook_end = 0, min(ms_to_samples(30_000, sr), track_len)
        errors.append(
            "engineer: blueprint has no 30s viral hook window; "
            "hook cut from the first 30 s instead"
        )
    hook_audio = mastered[:, hook_start:hook_end]
    sf.write(str(hook_path), hook_audio.T, sr, subtype="PCM_24")
    hook_seconds = hook_audio.shape[1] / sr

    # --- Audit trail for the Critic agent. ---------------------------------
    params: dict[str, Any] = dict(incoming_params)
    params.update(
        {
            "target_lufs": target_lufs,
            "club_master": club_master,
            "sidechain_depth_db": sidechain_depth_db,
            "carve_reduction_db": carve_reduction_db,
            "crossfade_ms": crossfade_ms,
            "sections_processed": section_records,
            "stem_roles": dict(roles),
            "missing_stems": missing,
            "kick_triggers": len(kick_triggers),
            "pre_master_clipped_samples": pre_clip["clipped_samples"],
            "limiter_applied": bool(limiter_info["limiter_applied"]),
            "limiter_backend": str(limiter_info["limiter_backend"]),
            "limiter_threshold_db": limiter_info["limiter_threshold_db"],
            "limiter_iterations": int(limiter_info["limiter_iterations"]),
            "limiter_ceiling_db": limiter_ceiling_db,
            "effective_ceiling_dbtp": float(
                limiter_info["effective_ceiling_dbtp"]
            ),
            "bass_gain_db": bass_gain_db,
            "beat_shift_ms": beat_shift_ms,
            "beat_shift_samples": int(round(beat_shift_ms * sr / 1000.0)),
            "lufs_shortfall_db": round(float(lufs_shortfall), 2),
            "measured_lufs": round(float(measured["integrated_lufs"]), 2),
            "measured_true_peak_dbtp": round(float(measured["true_peak_dbtp"]), 2),
            "measured_peak_dbfs": round(float(measured["peak_dbfs"]), 2),
            "written_lufs": round(float(written_metrics["integrated_lufs"]), 2),
            "written_true_peak_dbtp": round(
                float(written_metrics["true_peak_dbtp"]), 2
            ),
            "written_samplerate": int(written_sr),
            "written_subtype": str(written_info.subtype),
            "written_channels": int(written_info.channels),
            "master_path": str(master_path),
            "hook_path": str(hook_path),
            "hook_start_ms": (
                hook_30.start_ms if hook_30 is not None else 0
            ),
            "hook_duration_s": round(hook_seconds, 3),
            "sample_rate": sr,
            "output_dir": str(output_dir),
            "run_id": run_id,
        }
    )

    update: dict[str, Any] = {
        "master_path": str(master_path),
        "mix_params": params,
        "hook_path": str(hook_path),
    }
    if errors:
        update["errors"] = errors
    return update
