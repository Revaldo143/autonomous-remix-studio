"""DSP Mastering MCP Server — Autonomous Multi-Agent AI Music Producer & Remix Studio.

Exposes the surgical DSP core (``core.dsp_utils``) as Model Context Protocol
tools so the LangGraph agents (director / producer / engineer / critic) can
drive beat alignment, sidechain pumping, dynamic vocal carving, formant-preserving
time-stretch, binaural panning, loudness analysis, Pedalboard effect chains and
full mastering over file paths — never raw audio arrays over MCP.

Transport: stdio (``python mcp_servers/dsp_mastering_server.py``).
Project standard sample rate: 44_100 Hz. All file I/O goes through ``soundfile``;
outputs are written as 24-bit WAV (``subtype='PCM_24'``) unless noted otherwise.

VRAM hygiene
------------
This server loads **no** heavy models at all — DSP only, NumPy/SciPy level.
No ``torch`` import lives here on purpose. The generative servers
(``generative_audio_server.py``) that load RoFormer / WhisperX / MusicGen /
Seed-VC must instead load models sequentially and call
``torch.cuda.empty_cache()`` after each unload so a single 16 GB consumer GPU
(Colab T4 / A100) stays within budget. Keep that discipline out of this file.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Import shim: make `core.dsp_utils` importable whether this file is run as a
# script (`python mcp_servers/dsp_mastering_server.py`, sys.path[0] is
# mcp_servers/) or as a module from the project root. The DSP functions
# themselves are NOT redefined here — they live in core/dsp_utils.py and are
# imported from there, per the dependency contract.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.dsp_utils import (  # noqa: E402
    align_to_beat_grid,
    binaural_pan,
    detect_clipping,
    dynamic_vocal_carve,
    kick_triggered_sidechain,
    measure_loudness,
    ms_to_samples,
    normalize_to_lufs,
    samples_to_ms,
    time_stretch_formant_preserving,
)

mcp = FastMCP("dsp-mastering")

# Hard true-peak ceiling applied by master_track (dBTP). Never exceeded.
MASTER_CEILING_DBTP = -1.0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _f(value: Any, default: float = 0.0) -> float:
    """Coerce a possibly-numpy scalar to a plain JSON-safe float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _metric(metrics: dict[str, Any], *names: str, default: float = 0.0) -> float:
    """Read a loudness/clipping metric tolerating several key spellings.

    ``core.dsp_utils`` returns plain dicts; key casing is not part of the
    contract, so try each candidate in order (e.g. ``"lufs"``, ``"LUFS"``,
    ``"integrated_lufs"``).
    """
    for name in names:
        if name in metrics:
            return _f(metrics[name], default)
    lowered = {str(k).lower(): v for k, v in metrics.items()}
    for name in names:
        if name.lower() in lowered:
            return _f(lowered[name.lower()], default)
    return default


def _round_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable copy with floats rounded to 3 decimals."""
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            out[key] = value
        elif isinstance(value, (int, float, np.floating, np.integer)):
            out[key] = round(float(value), 3)
        elif isinstance(value, dict):
            out[key] = _round_metrics(value)
        elif isinstance(value, (list, tuple)):
            out[key] = [round(float(v), 3) if isinstance(v, (int, float, np.floating, np.integer)) else v for v in value]
        else:
            out[key] = value
    return out


def _read_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a WAV file as float64 ``(samples,)`` mono or ``(samples, channels)``."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    audio, sr = sf.read(str(file_path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float64)
    return audio, int(sr)


def _write_wav(path: str, audio: np.ndarray, sr: int) -> str:
    """Write float audio as 24-bit WAV, creating parent directories as needed."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), np.asarray(audio, dtype=np.float64), sr, subtype="PCM_24")
    return str(out_path)


def _audio_info(audio: np.ndarray, sr: int) -> dict[str, Any]:
    """Basic shape/duration facts shared by several tools."""
    channels = 1 if audio.ndim == 1 else int(audio.shape[1])
    num_samples = int(audio.shape[0])
    return {
        "sample_rate": int(sr),
        "channels": channels,
        "num_samples": num_samples,
        "duration_seconds": round(num_samples / sr, 3),
        "duration_ms": samples_to_ms(num_samples, sr),
    }


def _to_core(audio: np.ndarray) -> np.ndarray:
    """Convert soundfile layout to the DSP-core layout.

    ``soundfile`` reads audio as ``(samples,)`` mono or
    ``(samples, channels)``; ``core.dsp_utils`` works in ``(samples,)`` mono
    or ``(channels, samples)``. Mono is identical in both; stereo must be
    transposed. Every core call in this module goes through this helper so a
    stereo file is never silently misread as (samples=2, channels=N).
    """
    a = np.asarray(audio)
    if a.ndim == 2:
        return np.ascontiguousarray(a.T)
    return a


def _from_core(audio: np.ndarray) -> np.ndarray:
    """Convert the DSP-core layout back to the soundfile layout."""
    a = np.asarray(audio)
    if a.ndim == 2:
        return np.ascontiguousarray(a.T)
    return a


def _loudness_block(audio: np.ndarray, sr: int) -> dict[str, Any]:
    """Loudness + clipping metrics, JSON-safe, from the DSP core.

    ``audio`` is in soundfile layout; it is transposed to the core layout
    before measurement.
    """
    loudness = measure_loudness(_to_core(audio), sr)
    clipping = detect_clipping(_to_core(audio))
    return {
        "lufs": _metric(loudness, "lufs", "LUFS", "integrated_lufs"),
        "dbtp": _metric(loudness, "dbtp", "dBTP", "true_peak_dbtp", "true_peak"),
        "peak_dbfs": _metric(loudness, "peak_dbfs", "peak", "peak_dBFS", "max_peak_dbfs"),
        "rms_dbfs": _metric(loudness, "rms_dbfs", "rms_db", "rms", "rms_dBFS"),
        "clipping": _round_metrics(clipping) if isinstance(clipping, dict) else clipping,
    }


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------
@mcp.tool()
def align_beat_grid(
    audio_path: str,
    bpm: float,
    downbeat_ms_list: list[float],
    output_path: str,
) -> dict[str, Any]:
    """Align an audio file to the beat grid (sample-accurate).

    Downbeat timestamps are supplied in milliseconds and converted to exact
    sample indices at the file's native sample rate (project standard
    44_100 Hz) before the core aligns every beat to the grid implied by
    ``bpm``. Use this after stem separation so generated layers, drops and
    vocal chops land exactly on the beat instead of drifting by a few ms.

    Args:
        audio_path: Input WAV file.
        bpm: Tempo in beats per minute (must be > 0).
        downbeat_ms_list: Downbeat positions in milliseconds.
        output_path: Destination WAV file (24-bit).

    Returns:
        Output path, bpm, downbeat sample indices, and output audio info.
    """
    if bpm <= 0:
        raise ValueError(f"bpm must be > 0, got {bpm}")
    audio, sr = _read_wav(audio_path)
    downbeat_samples = [ms_to_samples(ms, sr) for ms in downbeat_ms_list]
    aligned = _from_core(align_to_beat_grid(_to_core(audio), sr, bpm, downbeat_samples))
    written = _write_wav(output_path, aligned, sr)
    return {
        "output_path": written,
        "bpm": float(bpm),
        "downbeat_ms_list": [float(ms) for ms in downbeat_ms_list],
        "downbeat_samples": [int(s) for s in downbeat_samples],
        **_audio_info(aligned, sr),
    }


@mcp.tool()
def apply_sidechain(
    input_path: str,
    kick_times_ms_list: list[float],
    output_path: str,
    depth_db: float = 6.0,
    release_ms: float = 120.0,
) -> dict[str, Any]:
    """Apply kick-triggered sidechain compression (the pumping effect).

    Every kick hit in ``kick_times_ms_list`` ducks the input (typically a bass
    or pad stem) by ``depth_db`` with a fast attack and a ``release_ms``
    recovery, then the gain rides back up — the classic club pump. Kick times
    are converted from milliseconds to exact sample indices first.

    Args:
        input_path: Input WAV file (usually bass / pad / full instrumental).
        kick_times_ms_list: Kick-drum hit positions in milliseconds.
        output_path: Destination WAV file (24-bit).
        depth_db: Maximum gain reduction in dB (>= 0).
        release_ms: Gain-recovery time in milliseconds (> 0).

    Returns:
        Output path, applied parameters, and before/after peak levels.
    """
    if depth_db < 0:
        raise ValueError(f"depth_db must be >= 0, got {depth_db}")
    if release_ms <= 0:
        raise ValueError(f"release_ms must be > 0, got {release_ms}")
    audio, sr = _read_wav(input_path)
    peak_before = _f(np.max(np.abs(audio))) if audio.size else 0.0
    kick_samples = [ms_to_samples(ms, sr) for ms in kick_times_ms_list]
    pumped = _from_core(
        kick_triggered_sidechain(
            _to_core(audio),
            kick_samples,
            sr,
            depth_db=depth_db,
            release_ms=release_ms,
        )
    )
    written = _write_wav(output_path, pumped, sr)
    peak_after = _f(np.max(np.abs(pumped))) if pumped.size else 0.0
    return {
        "output_path": written,
        "depth_db": float(depth_db),
        "release_ms": float(release_ms),
        "num_kicks": len(kick_times_ms_list),
        "peak_dbfs_before": round(float(20 * np.log10(peak_before)) if peak_before > 0 else -120.0, 3),
        "peak_dbfs_after": round(float(20 * np.log10(peak_after)) if peak_after > 0 else -120.0, 3),
        **_audio_info(pumped, sr),
    }


@mcp.tool()
def carve_vocals(
    instrumental_path: str,
    vocal_path: str,
    output_path: str,
    bands: int = 8,
    reduction_db: float = 3.0,
) -> dict[str, Any]:
    """Dynamically carve vocal frequencies out of the instrumental stem.

    Splits the instrumental into ``bands`` spectral bands and ducks only the
    bands where the vocal is currently energetic, by up to ``reduction_db``.
    Unlike a static EQ cut this follows the vocal moment by moment, so the
    mix stays full when the singer pauses and the vocal never gets masked
    when they sing. Both inputs must share the same sample rate and length;
    the vocal is trimmed/padded to the instrumental length if needed.

    Args:
        instrumental_path: Instrumental / music-bed WAV file.
        vocal_path: Isolated vocal stem WAV file.
        output_path: Destination WAV file (24-bit, carved instrumental).
        bands: Number of spectral bands (>= 1).
        reduction_db: Maximum per-band gain reduction in dB (>= 0).

    Returns:
        Output path, applied parameters, and before/after loudness.
    """
    if bands < 1:
        raise ValueError(f"bands must be >= 1, got {bands}")
    if reduction_db < 0:
        raise ValueError(f"reduction_db must be >= 0, got {reduction_db}")
    instrumental, sr = _read_wav(instrumental_path)
    vocal, vocal_sr = _read_wav(vocal_path)
    if vocal_sr != sr:
        raise ValueError(
            f"Sample-rate mismatch: instrumental {sr} Hz vs vocal {vocal_sr} Hz"
        )
    n = int(instrumental.shape[0])
    if vocal.shape[0] != n:
        # Zero-pad (or trim) to the instrumental length. np.resize would
        # REPEAT a shorter vocal to fill the gap, carving phantom vocal
        # energy into the tail — silence must stay silence.
        if vocal.ndim == 1:
            padded = np.zeros(n, dtype=vocal.dtype)
            copy_len = min(n, vocal.shape[0])
            padded[:copy_len] = vocal[:copy_len]
            vocal = padded
        else:
            padded = np.zeros((n, vocal.shape[1]), dtype=vocal.dtype)
            copy_len = min(n, vocal.shape[0])
            padded[:copy_len] = vocal[:copy_len]
            vocal = padded
    carved = _from_core(
        dynamic_vocal_carve(_to_core(instrumental), _to_core(vocal), sr, bands=bands, reduction_db=reduction_db)
    )
    written = _write_wav(output_path, carved, sr)
    return {
        "output_path": written,
        "bands": int(bands),
        "reduction_db": float(reduction_db),
        "lufs_before": _metric(measure_loudness(_to_core(instrumental), sr), "lufs", "LUFS", "integrated_lufs"),
        "lufs_after": _metric(measure_loudness(_to_core(carved), sr), "lufs", "LUFS", "integrated_lufs"),
        **_audio_info(carved, sr),
    }


@mcp.tool()
def time_stretch(
    input_path: str,
    rate: float,
    output_path: str,
) -> dict[str, Any]:
    """Formant-preserving time-stretch (pitch stays, tempo changes).

    ``rate > 1.0`` speeds the audio up (shorter), ``rate < 1.0`` slows it down
    (longer). Formants are preserved so a vocal stretched to a new BPM still
    sounds like the same singer instead of a chipmunk / Darth Vader. Use this
    to lock a stem or a reference vocal to the arrangement BPM.

    Args:
        input_path: Input WAV file.
        rate: Stretch factor (> 0; 1.0 = unchanged).
        output_path: Destination WAV file (24-bit).

    Returns:
        Output path, rate, and input/output durations.
    """
    if rate <= 0:
        raise ValueError(f"rate must be > 0, got {rate}")
    audio, sr = _read_wav(input_path)
    stretched = _from_core(time_stretch_formant_preserving(_to_core(audio), sr, rate))
    written = _write_wav(output_path, stretched, sr)
    return {
        "output_path": written,
        "rate": float(rate),
        "duration_seconds_before": round(float(audio.shape[0] / sr), 3),
        **_audio_info(stretched, sr),
    }


@mcp.tool()
def spatial_pan(
    input_path: str,
    azimuth_deg: float,
    output_path: str,
) -> dict[str, Any]:
    """Binaural / constant-power spatial pan of a mono or stereo source.

    Places the sound in the stereo field at ``azimuth_deg``: -90 is hard left,
    0 is centre, +90 is hard right. Mono inputs become stereo; stereo inputs
    are rebalanced. Used to give each stem its own seat in the mix (e.g.
    backing harmonies wide, lead vocal centred).

    Args:
        input_path: Input WAV file.
        azimuth_deg: Pan angle in degrees, -90 (left) to +90 (right).
        output_path: Destination WAV file (24-bit, stereo).

    Returns:
        Output path, azimuth, and output audio info.
    """
    if not -90.0 <= azimuth_deg <= 90.0:
        raise ValueError(f"azimuth_deg must be within [-90, 90], got {azimuth_deg}")
    audio, sr = _read_wav(input_path)
    # binaural_pan always returns stereo (2, samples) in core layout;
    # _from_core turns it into (samples, 2) for soundfile.
    panned = _from_core(binaural_pan(_to_core(audio), sr, azimuth_deg))
    written = _write_wav(output_path, panned, sr)
    return {
        "output_path": written,
        "azimuth_deg": float(azimuth_deg),
        **_audio_info(panned, sr),
    }


@mcp.tool()
def analyze_audio(audio_path: str) -> dict[str, Any]:
    """Analyze an audio file: loudness, clipping, peak, duration (no output file).

    Returns integrated LUFS, true peak (dBTP), sample peak (dBFS), RMS, a
    clipping report and duration — the numbers the critic agent uses to decide
    whether a render passes or needs a corrective re-render.

    Args:
        audio_path: WAV file to analyze.

    Returns:
        Loudness metrics, clipping report, peak, duration and format info.
    """
    audio, sr = _read_wav(audio_path)
    peak = _f(np.max(np.abs(audio))) if audio.size else 0.0
    return {
        "audio_path": str(Path(audio_path)),
        **_audio_info(audio, sr),
        "peak_dbfs": round(float(20 * np.log10(peak)) if peak > 0 else -120.0, 3),
        **_loudness_block(audio, sr),
    }


@mcp.tool()
def master_track(
    input_path: str,
    reference_path: str | None = None,
    target_lufs: float = -14.0,
    output_path: str = "",
) -> dict[str, Any]:
    """Full mastering chain: normalize to target LUFS with a hard -1.0 dBTP ceiling.

    Chain: (1) if ``reference_path`` is given and Matchering 2.0 is installed,
    match the reference's EQ curve / stereo width first; (2) normalize to
    ``target_lufs`` (-14 LUFS streaming default, -9 LUFS club master);
    (3) enforce the -1.0 dBTP hard ceiling with gain reduction (the ceiling
    wins over the LUFS target if they conflict — safety first).
    If Matchering is not installed, the tool still succeeds via the loudness
    path and reports ``reference_matched: false`` plus a ``warnings`` entry —
    the reference match is skipped, never fatal.

    Args:
        input_path: Mix WAV file to master.
        reference_path: Optional reference-track WAV for Matchering 2.0.
        target_lufs: Target integrated loudness (-14.0 streaming, -9.0 club).
        output_path: Destination WAV file (24-bit). Required.

    Returns:
        Output path, before/after loudness metrics, reference-match status,
        warnings, and whether the ceiling was hit.
    """
    if not output_path:
        raise ValueError("output_path is required for master_track")
    audio, sr = _read_wav(input_path)
    metrics_before = _loudness_block(audio, sr)
    warnings: list[str] = []
    reference_matched = False

    processed = np.asarray(audio, dtype=np.float64)

    if reference_path:
        ref_file = Path(reference_path)
        if not ref_file.is_file():
            warnings.append(f"Reference file not found, skipped: {reference_path}")
        else:
            try:
                from matchering import Result, process  # type: ignore

                tmp_matched = str(Path(output_path).with_name(Path(output_path).stem + ".matchering-tmp.wav"))
                Path(tmp_matched).parent.mkdir(parents=True, exist_ok=True)
                process(
                    target=input_path,
                    reference=str(ref_file),
                    results=[Result(file=tmp_matched, subtype="PCM_24")],
                )
                if Path(tmp_matched).is_file():
                    processed, matched_sr = _read_wav(tmp_matched)
                    if matched_sr != sr:
                        raise ValueError(
                            f"Matchering output sample rate {matched_sr} Hz != input {sr} Hz"
                        )
                    reference_matched = True
                    Path(tmp_matched).unlink(missing_ok=True)
                else:
                    warnings.append("Matchering produced no output file; fell back to loudness path.")
            except ImportError:
                warnings.append(
                    "matchering is not installed — reference match skipped; "
                    "mastered via loudness path only (pip install matchering to enable)."
                )
            except Exception as exc:  # noqa: BLE001 — reference match must never fail mastering
                warnings.append(f"Matchering failed ({exc}); fell back to loudness path.")

    # (2) Normalize to the target loudness.
    processed = _from_core(normalize_to_lufs(_to_core(processed), sr, target_lufs))

    # (3) Hard -1.0 dBTP ceiling. True peak scales linearly with gain, so one
    # corrective gain pass is exact; re-measure afterwards to confirm.
    ceiling_hit = False
    post_metrics = _loudness_block(processed, sr)
    if post_metrics["dbtp"] > MASTER_CEILING_DBTP:
        ceiling_hit = True
        gain_db = MASTER_CEILING_DBTP - post_metrics["dbtp"]
        processed = processed * float(10 ** (gain_db / 20.0))
        post_metrics = _loudness_block(processed, sr)
        warnings.append(
            f"True peak exceeded the {MASTER_CEILING_DBTP} dBTP ceiling; "
            f"applied {gain_db:.2f} dB gain reduction (LUFS now {post_metrics['lufs']:.2f})."
        )

    written = _write_wav(output_path, processed, sr)
    metrics_after = _loudness_block(processed, sr)
    return {
        "output_path": written,
        "target_lufs": float(target_lufs),
        "ceiling_dbtp": MASTER_CEILING_DBTP,
        "ceiling_hit": ceiling_hit,
        "reference_matched": reference_matched,
        "reference_path": reference_path,
        "warnings": warnings,
        "metrics_before": _round_metrics(metrics_before),
        "metrics_after": _round_metrics(metrics_after),
        **_audio_info(processed, sr),
    }


@mcp.tool()
def apply_pedalboard_chain(
    input_path: str,
    output_path: str,
    compressor_threshold_db: float = -18.0,
    reverb_mix: float = 0.15,
    highpass_hz: float = 30.0,
) -> dict[str, Any]:
    """Apply a Spotify Pedalboard studio chain: high-pass, compressor, reverb, limiter.

    Chain order: HighpassFilter -> Compressor -> Reverb -> Limiter (-1.0 dBTP
    ceiling). The high-pass removes subsonic rumble, the compressor glues the
    mix, the reverb adds space, and the limiter catches stray peaks. Pedalboard
    works on ``(channels, samples)`` float32 audio, so the file is transposed
    in and out internally.

    Args:
        input_path: Input WAV file.
        output_path: Destination WAV file (24-bit).
        compressor_threshold_db: Compressor threshold in dB.
        reverb_mix: Reverb wet level 0.0 (dry) to 1.0 (fully wet).
        highpass_hz: High-pass cutoff frequency in Hz (>= 0; 0 disables).

    Returns:
        Output path, applied chain parameters, and before/after loudness.
    """
    if not 0.0 <= reverb_mix <= 1.0:
        raise ValueError(f"reverb_mix must be within [0, 1], got {reverb_mix}")
    if highpass_hz < 0:
        raise ValueError(f"highpass_hz must be >= 0, got {highpass_hz}")
    try:
        from pedalboard import (  # type: ignore
            Compressor,
            HighpassFilter,
            Limiter,
            Pedalboard,
            Reverb,
        )
    except ImportError as exc:
        raise ImportError(
            "Spotify Pedalboard is not installed — apply_pedalboard_chain cannot run. "
            "Install it with: pip install pedalboard"
        ) from exc

    audio, sr = _read_wav(input_path)
    mono = audio.ndim == 1
    board_input = np.asarray(audio, dtype=np.float32)
    if mono:
        board_input = board_input[np.newaxis, :]
    else:
        board_input = np.ascontiguousarray(board_input.T)

    plugins: list[Any] = []
    if highpass_hz > 0:
        plugins.append(HighpassFilter(cutoff_frequency_hz=float(highpass_hz)))
    plugins.append(
        Compressor(
            threshold_db=float(compressor_threshold_db),
            ratio=3.0,
            attack_ms=10.0,
            release_ms=100.0,
        )
    )
    plugins.append(
        Reverb(
            room_size=0.5,
            damping=0.5,
            wet_level=float(reverb_mix),
            dry_level=1.0 - float(reverb_mix),
            width=1.0,
        )
    )
    plugins.append(Limiter(threshold_db=MASTER_CEILING_DBTP, release_ms=100.0))

    board = Pedalboard(plugins)
    processed = np.asarray(board(board_input, sr), dtype=np.float64)
    if mono:
        processed = processed[0]
    else:
        processed = np.ascontiguousarray(processed.T)

    written = _write_wav(output_path, processed, sr)
    return {
        "output_path": written,
        "compressor_threshold_db": float(compressor_threshold_db),
        "compressor_ratio": 3.0,
        "reverb_mix": float(reverb_mix),
        "highpass_hz": float(highpass_hz),
        "limiter_threshold_db": MASTER_CEILING_DBTP,
        "lufs_before": _metric(measure_loudness(_to_core(audio), sr), "lufs", "LUFS", "integrated_lufs"),
        "lufs_after": _metric(measure_loudness(_to_core(processed), sr), "lufs", "LUFS", "integrated_lufs"),
        **_audio_info(processed, sr),
    }


def main() -> None:
    """Run the DSP mastering MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
