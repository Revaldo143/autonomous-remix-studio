"""Generative Audio MCP Server — Autonomous Multi-Agent AI Music Producer
& Remix Studio.

Exposes generative audio as Model Context Protocol tools so the LangGraph
Producer agent can request new tempo/key-locked material (beats, risers,
melodic loops) over file paths — never raw audio arrays over MCP.

Transport: stdio (``python mcp_servers/generative_audio_server.py``).
Project standard sample rate: 44_100 Hz. All file I/O goes through
``soundfile``; outputs are written as 24-bit WAV (``subtype='PCM_24'``)
at exact sample lengths — a 4-bar 120 BPM beat is exactly 8.000 s
(352800 samples), never off by a frame.

Engines (honest reporting — never pretend a synth pattern is model output)
----------------------------------------------------------------------------
Every tool resolves its engine at call time and reports it in the result:

* ``"musicgen-melody"`` — the real generative model
  (``facebook/musicgen-melody`` via HuggingFace ``transformers``,
  weights under ``models/musicgen-melody/``). Used only when torch +
  transformers import AND the weights are on disk; otherwise the tool
  falls back and says so in ``"engine"`` / ``"warnings"``.
* ``"dsp-fallback"`` — deterministic NumPy synthesis (tempo-locked
  kick/snare/hat patterns, filtered-noise riser, scale-quantized pluck
  melody). This is the same synthesis the Producer agent already uses
  as its fallback; it is real DSP, not a stub, and it is labelled as
  such so no caller ever mistakes it for model output.

VRAM discipline
---------------
At most ONE heavy model is ever resident in this process: the
MusicGen-Melody model is loaded once (lazily, on first model call) and
cached for the process lifetime; there is no second model to load, so
the sequential load/unload discipline reduces to "load once, keep one".
``core.config.model_session("MusicGen-Melody")`` wraps every generate
call so the CUDA cache is emptied afterwards on GPU machines, and
``unload_model()`` frees the model explicitly on request. On this
CPU-only VM the model runs in bfloat16 (halves the ~6 GiB fp32 weights
to ~3 GiB) via ``torch_dtype=torch.bfloat16``.

Environment shims (in-process only, no shared state)
-----------------------------------------------------
This VM's torch 2.15.0 is ahead of its companions: torchvision 0.29.2
is ABI-incompatible with it and torchaudio 2.15.0 is unpublished, yet
transformers imports both at module level for the MusicGen classes.
``_ensure_torchvision_compat()`` injects an enum-only torchvision stub
and ``_ensure_torchaudio_stub()`` a torch.stft-backed torchaudio stub
(Spectrogram + resample) when the real packages cannot import. Both
are no-ops where the real packages work, and both are real signal
processing where they stand in — the model, weights and audio are
genuine either way. One extra catch: transformers'
``is_torchaudio_available()`` ALSO checks
``importlib.metadata.version("torchaudio")``, so a module stub alone
is not enough — a minimal ``torchaudio-0.0.0.dist-info/METADATA``
lives in the project tree at ``models/_shims/`` and the stub adds
that directory to ``sys.path`` so the version lookup returns "0.0.0"
through stock importlib.metadata behaviour (no venv surgery).

Measured behaviour on this CPU-only VM (2026-11-17, 8 GB RAM, 4 CPU)
---------------------------------------------------------------------
The real model WORKS here — it is the active engine, not the fallback:
load ~30 s, then ~27 s of wall clock per second of generated audio
(2 s in 54 s, autoregressive, 50 tokens/s), peak RSS ~5.8 GB. The
model renders MONO, so the server up-mixes to stereo by duplicating
the channel (same as the DSP path's ``_stereo()``); generation is
stochastic (``do_sample=True``), so two calls with the same arguments
produce different audio — that is the model, not a bug. Two further
CPU/bf16 fixes live in ``_generate_with_musicgen``: generation runs
under ``torch.autocast`` (the processor's float32 chroma features
and the T5 encoder's float32 activations otherwise die on the first
bf16 Linear with a dtype error), and the model output is cast
``.float()`` before ``.numpy()`` (NumPy has no bf16 dtype).
Implication for callers: the Producer's 600 s MCP budget covers
roughly <= 20 s of model audio per call; longer requests (a 30 s
beat ≈ 14 min) will time out at the client and fall back to DSP —
honest, but slow. On a GPU the same model runs near real time.

Contract compatibility
------------------------
The Producer agent calls ``generate_beat(bpm, key, bars, style)`` and
``generate_riser(duration_ms, style)`` with NO output path; the tools
therefore accept ``key`` (used in the MusicGen prompt) and default
``output_path`` to ``outputs/generative/`` under the project root.
The task-facing signatures are ``generate_beat(bpm, bars, style,
output_path)``, ``generate_riser(duration_s, style, output_path)`` and
``generate_melody(bpm, bars, scale, output_path)`` — both calling
conventions are accepted.
"""

from __future__ import annotations

import gc
import hashlib
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Import shim: make `core.*` / `agents.state` importable whether this file
# is run as a script (`python mcp_servers/generative_audio_server.py`,
# sys.path[0] is mcp_servers/) or as a module from the project root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.state import SAMPLE_RATE  # noqa: E402
from core.config import MODEL_VRAM_GB, get_device, model_session  # noqa: E402
from core.dsp_utils import ms_to_samples, samples_to_ms  # noqa: E402

mcp = FastMCP("generative-audio")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: MusicGen's native sample rate (model output is 32 kHz stereo).
MUSICGEN_SR = 32000

#: MusicGen audio-token frame rate: 50 tokens per second of audio.
MUSICGEN_FRAME_RATE = 50

#: Local weights directory for facebook/musicgen-melody.
MUSICGEN_MODEL_ID = "facebook/musicgen-melody"
MUSICGEN_WEIGHTS_DIR = _PROJECT_ROOT / "models" / "musicgen-melody"

#: Default output directory when the caller passes no output_path.
DEFAULT_OUT_DIR = _PROJECT_ROOT / "outputs" / "generative"

#: Validation bounds (task contract).
MIN_BPM = 40.0
MAX_BPM = 240.0

#: Peak targets after normalization (linear amplitude).
BEAT_PEAK = 0.9
RISER_PEAK = 0.8
MELODY_PEAK = 0.85
# Minimum chroma in-scale fraction for a MusicGen melody to be kept.
# Chance is 7/12 (0.583) for a 7-note scale; the DSP melody (exact by
# construction) reads 0.756; three independent model runs read 0.595,
# 0.626, 0.758. A bar of 0.65 keeps clearly-in-key model loops and
# sends at-chance loops to the exact DSP fallback instead of shipping
# an out-of-key stem. It is a named constant so the parent can tune it.
MELODY_MIN_CHROMA = 0.65
# Minimum rise ratio (last-chunk RMS / first-chunk RMS, 8 chunks) for
# a MusicGen riser to be kept. The DSP riser reads ~400x; one model
# run read 0.83x (flat energy, not a riser at all). Same bar as the
# test uses, so a kept model riser passes it.
RISER_MIN_RISE_RATIO = 3.0

#: Model-generation defaults.
MUSICGEN_GUIDANCE_SCALE = 3.0

#: In-process cache: at most one model resident (see module docstring).
_MODEL_CACHE: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _f(value: Any, default: float = 0.0) -> float:
    """Coerce a possibly-numpy scalar to a plain JSON-safe float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable copy with floats rounded to 3 decimals."""
    out: dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, bool):
            out[key] = value
        elif isinstance(value, (int, float, np.floating, np.integer)):
            out[key] = round(float(value), 3)
        elif isinstance(value, dict):
            out[key] = _round_result(value)
        elif isinstance(value, (list, tuple)):
            out[key] = [
                round(float(v), 3)
                if isinstance(v, (int, float, np.floating, np.integer))
                else v
                for v in value
            ]
        else:
            out[key] = value
    return out


def _write_wav(path: str, audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    """Write float audio as 24-bit WAV, creating parent directories."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), np.asarray(audio, dtype=np.float64), sr, subtype="PCM_24")
    return str(out_path)


def _audio_info(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict[str, Any]:
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


def _default_output_path(kind: str) -> str:
    """Auto output path under outputs/generative/ (producer calls pass none)."""
    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    return str(DEFAULT_OUT_DIR / f"{kind}_{uuid.uuid4().hex[:8]}.wav")


def _exact_length(audio: np.ndarray, n_samples: int) -> np.ndarray:
    """Trim or zero-pad audio to EXACTLY ``n_samples`` (never resample).

    Model outputs and resampling can land a frame short/long; the studio
    contract is sample-exact, so the tail is trimmed or silence-padded.
    """
    a = np.asarray(audio)
    if a.shape[0] == n_samples:
        return a
    if a.shape[0] > n_samples:
        return np.ascontiguousarray(a[:n_samples])
    pad_shape = (n_samples - a.shape[0],) + a.shape[1:]
    return np.concatenate([a, np.zeros(pad_shape, dtype=a.dtype)], axis=0)


def _peak_normalize(audio: np.ndarray, peak: float) -> np.ndarray:
    """Scale audio so its absolute peak equals ``peak`` (no-op on silence)."""
    a = np.asarray(audio, dtype=np.float64)
    max_abs = float(np.max(np.abs(a))) if a.size else 0.0
    if max_abs <= 0.0:
        return a
    return a * (peak / max_abs)


def _stereo(mono: np.ndarray) -> np.ndarray:
    """Up-mix a mono signal to identical stereo (downstream consistency)."""
    m = np.asarray(mono, dtype=np.float64)
    return np.column_stack([m, m])


# ---------------------------------------------------------------------------
# Input validation (clean errors, per the task contract)
# ---------------------------------------------------------------------------
def _validate_bpm(bpm: float) -> float:
    try:
        bpm_f = float(bpm)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bpm must be a number, got {bpm!r}") from exc
    if not np.isfinite(bpm_f) or not MIN_BPM <= bpm_f <= MAX_BPM:
        raise ValueError(f"bpm must be within [{MIN_BPM}, {MAX_BPM}], got {bpm}")
    return bpm_f


def _validate_bars(bars: int) -> int:
    try:
        bars_i = int(bars)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bars must be an integer, got {bars!r}") from exc
    if bars_i < 1 or float(bars) != float(bars_i):
        raise ValueError(f"bars must be an integer >= 1, got {bars}")
    return bars_i


def _validate_style(style: Any) -> str:
    if not isinstance(style, str) or not style.strip():
        raise ValueError(f"style must be a non-empty string, got {style!r}")
    return style.strip()


def _validate_output_path(output_path: Any) -> str | None:
    if output_path is None:
        return None
    if not isinstance(output_path, str) or not output_path.strip():
        raise ValueError(f"output_path must be a non-empty string, got {output_path!r}")
    return output_path.strip()


def _samples_for_bars(bpm: float, bars: int) -> int:
    """Exact sample count for ``bars`` bars of 4/4 at ``bpm``.

    4 bars @ 120 BPM = 8.000 s = 352800 samples. Non-integer beat lengths
    (e.g. 126.05 BPM) round to the nearest sample — the closest possible
    to exact at 44 100 Hz.
    """
    seconds = bars * 4 * 60.0 / bpm
    return max(1, int(round(seconds * SAMPLE_RATE)))


def _samples_for_seconds(duration_s: float) -> int:
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError(f"duration must be > 0 seconds, got {duration_s}")
    return max(1, int(round(duration_s * SAMPLE_RATE)))


# ---------------------------------------------------------------------------
# Scale parsing (for generate_melody)
# ---------------------------------------------------------------------------
_NOTE_SEMITONES: dict[str, int] = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4,
    "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9,
    "A#": 10, "BB": 10, "B": 11,
}

#: Scale interval sets (semitones above the root).
_SCALE_INTERVALS: dict[str, list[int]] = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],  # natural minor
    "major pentatonic": [0, 2, 4, 7, 9],
    "minor pentatonic": [0, 3, 5, 7, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
}


def _parse_scale(scale: str) -> tuple[int, list[int], str]:
    """Parse e.g. "D major" / "A minor" / "C major pentatonic".

    Returns ``(root_midi, intervals, canonical_name)``. Raises ValueError
    naming the supported scales when the string cannot be parsed.
    """
    if not isinstance(scale, str) or not scale.strip():
        raise ValueError(f"scale must be a non-empty string, got {scale!r}")
    text = " ".join(scale.strip().split())
    # Longest mode name first so "major pentatonic" wins over "major".
    for mode in sorted(_SCALE_INTERVALS, key=len, reverse=True):
        if text.lower().endswith(mode):
            root_text = text[: -len(mode)].strip()
            break
    else:
        raise ValueError(
            f"unsupported scale {scale!r}; expected '<root> <mode>' with mode in "
            f"{sorted(_SCALE_INTERVALS)}"
        )
    root_key = root_text.upper().replace("♯", "#").replace("♭", "B")
    if root_key not in _NOTE_SEMITONES:
        raise ValueError(
            f"unknown root note {root_text!r} in scale {scale!r}; "
            f"expected one of {sorted(_NOTE_SEMITONES)}"
        )
    root_midi = 60 + _NOTE_SEMITONES[root_key]  # root in octave 4 (C4 = 60)
    return root_midi, _SCALE_INTERVALS[mode], f"{root_text} {mode}"


def _scale_pitch_classes(root_midi: int, intervals: list[int]) -> set[int]:
    """Pitch classes (0-11) of the scale, for chroma verification."""
    return {(root_midi + i) % 12 for i in intervals}


def _chroma_in_scale_fraction(audio: np.ndarray, pitch_classes: set[int]) -> float:
    """Fraction of chroma energy on in-scale pitch classes (librosa CQT).

    Chroma folds the spectrum into 12 pitch classes, so it is the
    right instrument for polyphonic model output (a per-step dominant
    pitch would just measure the loudest bass note). Chance level for
    a 7-note scale is 7/12 (0.583); leakage caps even a perfectly
    in-scale line at ~0.76 (the DSP melody, exact by construction,
    reads 0.756). Measured MusicGen-Melody loops, three independent
    runs: 0.595, 0.626, 0.758 — the model is in-key-ish but NOT
    scale-guaranteed, which is why generate_melody gates on this
    number (see MELODY_MIN_CHROMA) instead of trusting the model.
    librosa is imported lazily so the beat/riser paths never pay for
    it; the melody path already pays for torch/transformers.
    """
    import librosa

    mono = np.asarray(audio, dtype=np.float64)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    if mono.size == 0 or not np.any(mono):
        return 0.0
    chroma = librosa.feature.chroma_cqt(y=mono, sr=SAMPLE_RATE)
    mean_chroma = chroma.mean(axis=1)
    total = float(np.sum(mean_chroma))
    if total <= 0:
        return 0.0
    return float(sum(mean_chroma[pc] for pc in pitch_classes) / total)


def _rise_ratio(audio: np.ndarray, n_chunks: int = 8) -> float:
    """Last-chunk RMS / first-chunk RMS (NumPy only) — the riser check.

    A true riser builds energy: the DSP riser reads ~400x. A model
    output that starts loud and stays flat reads ~1x or less and is
    not a riser at all, however good it sounds. Returns 0.0 for
    silence (first chunk RMS 0).
    """
    mono = np.asarray(audio, dtype=np.float64)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    if mono.size == 0 or not np.any(mono):
        return 0.0
    chunks = np.array_split(mono, n_chunks)
    rms = [float(np.sqrt(np.mean(c**2))) if c.size else 0.0 for c in chunks]
    if rms[0] <= 0:
        return 0.0
    return rms[-1] / rms[0]


# ---------------------------------------------------------------------------
# DSP synthesis — deterministic, tempo-locked, sample-exact
# ---------------------------------------------------------------------------
def _synth_kick_hit(sr: int) -> np.ndarray:
    """One kick hit: sine burst with a 150 -> 50 Hz pitch envelope and
    ~45 ms exponential decay (mirrors producer_agent's kick)."""
    burst_len = ms_to_samples(150, sr)
    t = np.arange(burst_len, dtype=np.float64) / sr
    freq = 50.0 + 100.0 * np.exp(-t / 0.03)
    phase = 2.0 * np.pi * np.cumsum(freq) / sr
    return np.sin(phase) * np.exp(-t / 0.045)


def _synth_snare_hit(sr: int) -> np.ndarray:
    """Snare: 180 Hz body + high-passed noise burst, ~120 ms."""
    n = ms_to_samples(120, sr)
    t = np.arange(n, dtype=np.float64) / sr
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(n)
    # Crude high-pass: first difference removes the DC/low body.
    noise_hp = np.concatenate([[0.0], np.diff(noise)])
    body = np.sin(2.0 * np.pi * 180.0 * t) * np.exp(-t / 0.05)
    return 0.6 * body + 0.5 * noise_hp * np.exp(-t / 0.06)


def _synth_hat_hit(sr: int, open_hat: bool = False) -> np.ndarray:
    """Hi-hat: high-passed noise, 40 ms (closed) / 180 ms (open)."""
    n = ms_to_samples(180 if open_hat else 40, sr)
    rng = np.random.default_rng(11)
    noise = rng.standard_normal(n)
    noise_hp = np.concatenate([[0.0], np.diff(noise)])
    decay = 0.09 if open_hat else 0.02
    t = np.arange(n, dtype=np.float64) / sr
    return noise_hp * np.exp(-t / decay)


def _synth_clap_hit(sr: int) -> np.ndarray:
    """Clap: three 10 ms noise bursts 12 ms apart, ~150 ms tail."""
    n = ms_to_samples(150, sr)
    rng = np.random.default_rng(13)
    out = np.zeros(n, dtype=np.float64)
    for k in range(3):
        start = ms_to_samples(12 * k, sr)
        burst_len = ms_to_samples(10, sr)
        out[start : start + burst_len] += rng.standard_normal(burst_len)
    t = np.arange(n, dtype=np.float64) / sr
    return out * np.exp(-t / 0.05)


def _synth_cowbell_hit(sr: int) -> np.ndarray:
    """Phonk cowbell: detuned 587 + 845 Hz sines, ~250 ms decay."""
    n = ms_to_samples(250, sr)
    t = np.arange(n, dtype=np.float64) / sr
    return (
        np.sin(2.0 * np.pi * 587.0 * t) + 0.7 * np.sin(2.0 * np.pi * 845.0 * t)
    ) * np.exp(-t / 0.08)


#: Style profiles: 16-step (16th-note) patterns per bar. Steps are 0-15
#: within one bar; a step list per instrument. "swing" delays off-beat
#: 16ths (odd steps) by a fraction of one step. Unknown styles fall back
#: to "electronic" (four-on-the-floor) and say so in the result.
_STYLE_PROFILES: dict[str, dict[str, Any]] = {
    "electronic": {
        "kick": [0, 4, 8, 12],
        "snare": [4, 12],
        "clap": [],
        "hat": [2, 6, 10, 14],
        "open_hat": [14],
        "cowbell": [],
        "swing": 0.0,
    },
    "kuthu": {  # dappankuthu: driving 4-on-floor + dense offbeat hats
        "kick": [0, 4, 8, 12],
        "snare": [4, 12],
        "clap": [4, 12],
        "hat": [2, 6, 10, 14, 3, 7, 11, 15],
        "open_hat": [14],
        "cowbell": [],
        "swing": 0.0,
    },
    "phonk": {  # punchy kick, snare on 2 & 4, cowbell offbeats, 16th hats
        "kick": [0, 4, 8, 12, 10],
        "snare": [4, 12],
        "clap": [],
        "hat": [0, 2, 4, 6, 8, 10, 12, 14, 15],
        "open_hat": [],
        "cowbell": [2, 6, 10, 14],
        "swing": 0.0,
    },
    "trap": {  # half-time: snare on beat 3, sparse syncopated kicks, hat rolls
        "kick": [0, 7, 10],
        "snare": [8],
        "clap": [],
        "hat": [0, 2, 4, 6, 8, 10, 12, 14, 3, 11],
        "open_hat": [14],
        "cowbell": [],
        "swing": 0.0,
    },
    "lo-fi": {  # soft kicks, swung hats, snare on 2 & 4
        "kick": [0, 8, 10],
        "snare": [4, 12],
        "clap": [],
        "hat": [2, 6, 10, 14],
        "open_hat": [],
        "cowbell": [],
        "swing": 0.3,
    },
}


def _resolve_style_profile(style: str) -> tuple[str, dict[str, Any]]:
    """Pick a style profile by substring match; unknown -> electronic."""
    lowered = style.lower()
    for name, profile in _STYLE_PROFILES.items():
        if name in lowered:
            return name, profile
    return "electronic", _STYLE_PROFILES["electronic"]


def _synthesize_beat_dsp(
    bpm: float,
    bars: int,
    style: str,
    sr: int = SAMPLE_RATE,
) -> tuple[np.ndarray, str]:
    """DSP beat: tempo-locked drum pattern, exact length, stereo.

    Returns ``(stereo_audio, resolved_style)``. Kicks land on the beat
    grid by construction (step 0 of every bar group is a downbeat), so a
    low-band onset detector sees real energy on every beat.
    """
    n_samples = _samples_for_bars(bpm, bars)
    profile_name, profile = _resolve_style_profile(style)
    step_len = int(round(sr * 60.0 / bpm / 4.0))  # one 16th note, samples
    steps_per_bar = 16
    swing = float(profile["swing"])

    hits = {
        "kick": _synth_kick_hit(sr),
        "snare": _synth_snare_hit(sr),
        "clap": _synth_clap_hit(sr),
        "hat": _synth_hat_hit(sr, open_hat=False),
        "open_hat": _synth_hat_hit(sr, open_hat=True),
        "cowbell": _synth_cowbell_hit(sr),
    }
    # Per-instrument mix gains and stereo pan (-1 left, +1 right).
    mix = {
        "kick": (1.0, 0.0),
        "snare": (0.7, 0.0),
        "clap": (0.5, 0.0),
        "hat": (0.35, 0.4),
        "open_hat": (0.3, -0.4),
        "cowbell": (0.4, 0.3),
    }
    left = np.zeros(n_samples, dtype=np.float64)
    right = np.zeros(n_samples, dtype=np.float64)
    total_steps = bars * steps_per_bar
    for step in range(total_steps):
        bar_step = step % steps_per_bar
        # Swing: delay odd 16th steps by a fraction of one step.
        offset = int(round(swing * step_len)) if (bar_step % 2 == 1) else 0
        start = step * step_len + offset
        if start >= n_samples:
            continue
        for name, steps in profile.items():
            if name in ("swing",):
                continue
            if bar_step not in steps:
                continue
            hit = hits[name]
            gain, pan = mix[name]
            end = min(n_samples, start + len(hit))
            seg = hit[: end - start] * gain
            # Constant-power pan.
            left[start:end] += seg * float(np.sqrt((1.0 - pan) / 2.0) * np.sqrt(2.0))
            right[start:end] += seg * float(np.sqrt((1.0 + pan) / 2.0) * np.sqrt(2.0))
    stereo = np.column_stack([left, right])
    return _peak_normalize(stereo, BEAT_PEAK), profile_name


def _synthesize_riser_dsp(
    duration_s: float,
    style: str,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """DSP riser: white noise through a time-varying one-pole low-pass
    sweeping 300 Hz -> 9 kHz under a squared rising amplitude ramp, plus
    a quiet sine sweep (200 -> 1200 Hz) for tonal lift. Deterministic
    (seed 0); peak-normalized to 0.8. Energy rises monotonically in
    practice: both the filter opening and the ramp push RMS up."""
    n = _samples_for_seconds(duration_s)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(n)
    fc = np.geomspace(300.0, 9000.0, n)
    alpha = 1.0 - np.exp(-2.0 * np.pi * fc / sr)
    out = np.empty(n, dtype=np.float64)
    y = 0.0
    for i in range(n):  # one-pole filter must run sequentially
        y += alpha[i] * (noise[i] - y)
        out[i] = y
    ramp = np.linspace(0.0, 1.0, n) ** 2
    out = out * ramp
    # Tonal sweep layered underneath.
    t = np.arange(n, dtype=np.float64) / sr
    freq = 200.0 * (1200.0 / 200.0) ** (t / duration_s)
    phase = 2.0 * np.pi * np.cumsum(freq) / sr
    out = out + 0.25 * np.sin(phase) * ramp
    return _peak_normalize(_stereo(out), RISER_PEAK)


def _synthesize_melody_dsp(
    bpm: float,
    bars: int,
    scale: str,
    style: str,
    sr: int = SAMPLE_RATE,
) -> tuple[np.ndarray, str]:
    """DSP melody: deterministic random-walk pluck melody quantized to
    ``scale`` (8th-note steps, some rests), with a root bass note on each
    bar start. Returns ``(stereo_audio, canonical_scale)``. Every note
    pitch is in-scale by construction, so chroma stays in the scale.
    """
    n_samples = _samples_for_bars(bpm, bars)
    root_midi, intervals, canonical = _parse_scale(scale)
    # Two octaves of scale degrees above the root (MIDI 60..~84).
    scale_notes: list[int] = []
    for octave in range(0, 3):
        for interval in intervals:
            midi = root_midi + 12 * octave + interval
            if 55 <= midi <= 86:
                scale_notes.append(midi)
    scale_notes.sort()
    root_idx = scale_notes.index(root_midi)

    # Deterministic walk: seed from a STABLE hash of scale + bpm so the
    # same call replays identically in every process (builtin hash() is
    # salted per process and would give a different melody each run).
    seed = int.from_bytes(
        hashlib.sha256(f"{canonical}|{bpm:.2f}".encode()).digest()[:4], "little"
    ) % (2**31)
    rng = np.random.default_rng(seed)
    step_len = int(round(sr * 60.0 / bpm / 2.0))  # one 8th note
    total_steps = bars * 8
    mono = np.zeros(n_samples, dtype=np.float64)
    idx = root_idx + 4  # start a fifth-ish above the root
    for step in range(total_steps):
        if step % 8 == 0:
            idx = root_idx + 4  # re-anchor at each bar start
        # Random walk with a pull toward the middle of the range.
        move = int(rng.choice([-2, -1, -1, 0, 1, 1, 2]))
        idx = int(np.clip(idx + move, 0, len(scale_notes) - 1))
        if rng.random() < 0.15:
            continue  # rest
        midi = scale_notes[idx]
        freq = 440.0 * 2.0 ** ((midi - 69) / 12.0)
        start = step * step_len
        if start >= n_samples:
            break
        note_len = min(step_len * (2 if rng.random() < 0.25 else 1), n_samples - start)
        t = np.arange(note_len, dtype=np.float64) / sr
        pluck = (
            np.sin(2.0 * np.pi * freq * t)
            + 0.3 * np.sin(2.0 * np.pi * 2.0 * freq * t)
            + 0.1 * np.sin(2.0 * np.pi * 3.0 * freq * t)
        ) * np.exp(-t / 0.22)
        mono[start : start + note_len] += pluck
        if step % 8 == 0:
            # Root bass note (one octave below) on each bar start.
            bass_freq = 440.0 * 2.0 ** ((root_midi - 12 - 69) / 12.0)
            bass_len = min(step_len * 4, n_samples - start)
            tb = np.arange(bass_len, dtype=np.float64) / sr
            mono[start : start + bass_len] += (
                0.5 * np.sin(2.0 * np.pi * bass_freq * tb) * np.exp(-tb / 0.5)
            )
    stereo = _stereo(mono)
    return _peak_normalize(stereo, MELODY_PEAK), canonical


# ---------------------------------------------------------------------------
# MusicGen model path (real generative engine)
# ---------------------------------------------------------------------------
def _ensure_torchvision_compat() -> None:
    """Work around a broken torchvision in this venv.

    The venv has torch 2.15.0 (pinned) but torchvision 0.29.2, which was
    built for torch 2.14 — its C++ ops fail to register
    (``RuntimeError: operator torchvision::nms does not exist``) and the
    import raises. transformers' ``image_utils`` imports
    ``torchvision.transforms.InterpolationMode`` at module level, so ANY
    transformers model import (including the audio-only MusicGen) dies
    with it — even though the audio model never uses torchvision.
    torchvision 0.30 (the torch 2.15 match) is not published yet, and
    uninstalling torchvision is not an option (a sibling server's
    onnx2torch declares it), so when the real import fails this injects
    a minimal in-process stub providing only that enum. A no-op when
    the real torchvision imports cleanly (e.g. on a GPU box with a
    matching pair). The stub lives only in this server process; it
    changes no shared state and never affects the model or the audio.
    """
    try:
        import torchvision  # noqa: F401

        return
    except Exception:  # noqa: BLE001 — any import failure -> stub it
        pass
    import enum
    import types

    for name in [m for m in list(sys.modules) if m == "torchvision" or m.startswith("torchvision.")]:
        del sys.modules[name]

    class InterpolationMode(enum.Enum):
        NEAREST = "nearest"
        NEAREST_EXACT = "nearest_exact"
        BILINEAR = "bilinear"
        BICUBIC = "bicubic"
        BOX = "box"
        HAMMING = "hamming"
        LANCZOS = "lanczos"

    import enum
    import importlib.machinery
    import types

    def _stub_module(name: str, package: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        mod.__package__ = package
        return mod

    # transformers' availability checks call importlib.util.find_spec(),
    # which raises ValueError when a loaded module has __spec__ None —
    # every stub module below gets a spec (and the package an empty
    # __path__) so those checks pass.
    transforms_mod = _stub_module("torchvision.transforms", "torchvision")
    transforms_mod.InterpolationMode = InterpolationMode
    # image_processing_utils_fast.py imports torchvision.transforms.v2
    # (gated behind is_torchvision_available()); the functional submodule
    # is import-only here — audio never calls it.
    v2_mod = _stub_module("torchvision.transforms.v2", "torchvision.transforms")
    functional_mod = _stub_module(
        "torchvision.transforms.v2.functional", "torchvision.transforms.v2"
    )
    v2_mod.functional = functional_mod
    transforms_mod.v2 = v2_mod
    # video_utils.py does `from torchvision import io as torchvision_io`
    # and only calls read_video for video files — never in the audio
    # path. Provide the names so the import succeeds; calling them is
    # an honest error.
    io_mod = _stub_module("torchvision.io", "torchvision")

    def _video_unavailable(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "torchvision is unusable in this venv (enum-only shim); "
            "video I/O is not available"
        )

    io_mod.read_video = _video_unavailable
    io_mod.write_video = _video_unavailable
    tv_mod = _stub_module("torchvision", "")
    tv_mod.transforms = transforms_mod
    tv_mod.io = io_mod
    tv_mod.__version__ = "0.0.0-stub (torchvision unusable; enum-only shim)"
    tv_mod.__path__ = []
    sys.modules["torchvision"] = tv_mod
    sys.modules["torchvision.transforms"] = transforms_mod
    sys.modules["torchvision.transforms.v2"] = v2_mod
    sys.modules["torchvision.transforms.v2.functional"] = functional_mod
    sys.modules["torchvision.io"] = io_mod


def _ensure_torchaudio_stub() -> None:
    """Provide a minimal in-process torchaudio when the real package is
    absent.

    torchaudio 2.15.0 (the match for the pinned torch 2.15.0) is not
    published anywhere — the latest release is 2.11.1, built for torch
    2.11 and ABI-incompatible with torch 2.15.0 — so the real package
    cannot be installed here at all. The MusicGen-Melody feature
    extractor needs torchaudio for exactly two things:
    ``transforms.Spectrogram`` (power spectrogram for the chroma
    conditioning) and ``functional.resample`` (only when the input
    sample rate differs from 32 kHz — this server always passes 32 kHz
    audio, so it never fires). The stub implements Spectrogram with
    ``torch.stft`` — the same backend real torchaudio 2.x uses — and
    resample with ``scipy.signal.resample_poly``. Real signal
    processing, not a fake: the chroma conditioning the model receives
    is computed for real. A no-op when real torchaudio imports.
    Lives only in this server process; changes no shared state.
    """
    try:
        import torchaudio  # noqa: F401

        return
    except Exception:  # noqa: BLE001 — any import failure -> stub it
        pass
    import importlib.machinery
    import types

    import torch
    from scipy.signal import resample_poly

    class _Spectrogram(torch.nn.Module):
        """torch.stft-backed drop-in for torchaudio.transforms.Spectrogram.

        Matches the constructor kwargs the MusicGen feature extractor
        passes (n_fft, win_length, hop_length, power, center, pad,
        normalized) and the forward semantics (Hann window, centred
        STFT, |STFT|**power).
        """

        def __init__(
            self,
            n_fft: int = 400,
            win_length: int | None = None,
            hop_length: int | None = None,
            power: float = 2.0,
            center: bool = True,
            pad: int = 0,
            normalized: bool = True,
        ) -> None:
            super().__init__()
            self.n_fft = n_fft
            self.win_length = win_length if win_length is not None else n_fft
            self.hop_length = (
                hop_length if hop_length is not None else self.win_length // 2
            )
            self.power = power
            self.center = center
            self.pad = pad
            self.normalized = normalized

        def forward(self, waveform: torch.Tensor) -> torch.Tensor:
            window = torch.hann_window(
                self.win_length, device=waveform.device, dtype=waveform.dtype
            )
            if self.pad > 0:
                waveform = torch.nn.functional.pad(
                    waveform, (self.pad, self.pad), mode="constant", value=0.0
                )
            spec = torch.stft(
                waveform,
                self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                center=self.center,
                pad_mode="constant",
                normalized=self.normalized,
                onesided=True,
                return_complex=True,
            )
            magnitude = spec.abs()
            if self.power == 1.0:
                return magnitude
            return magnitude**self.power

    def _resample(
        waveform: torch.Tensor,
        orig_freq: int,
        new_freq: int,
        rolloff: float = 0.99,
        lowpass_filter_width: int = 6,
    ) -> torch.Tensor:
        """Rational resampling via scipy (only called when sample rates
        differ; this server always passes 32 kHz audio, so it is a
        safety net, not the hot path)."""
        if orig_freq == new_freq:
            return waveform
        a = waveform.detach().cpu().numpy()
        gcd = np.gcd(int(orig_freq), int(new_freq))
        up, down = int(new_freq) // gcd, int(orig_freq) // gcd
        out = resample_poly(a, up, down, axis=-1)
        return torch.as_tensor(out, dtype=waveform.dtype, device=waveform.device)

    def _stub_module(name: str, package: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        mod.__package__ = package
        return mod

    for name in [m for m in list(sys.modules) if m == "torchaudio" or m.startswith("torchaudio.")]:
        del sys.modules[name]
    transforms_mod = _stub_module("torchaudio.transforms", "torchaudio")
    transforms_mod.Spectrogram = _Spectrogram
    functional_mod = _stub_module("torchaudio.functional", "torchaudio")
    functional_mod.resample = _resample
    ta_mod = _stub_module("torchaudio", "")
    ta_mod.transforms = transforms_mod
    ta_mod.functional = functional_mod
    ta_mod.__version__ = "0.0.0-stub (torchaudio unpublished for torch 2.15; stft-backed shim)"
    ta_mod.__path__ = []
    sys.modules["torchaudio"] = ta_mod
    sys.modules["torchaudio.transforms"] = transforms_mod
    sys.modules["torchaudio.functional"] = functional_mod
    # transformers' is_torchaudio_available() ALSO checks
    # importlib.metadata.version("torchaudio") — a module stub alone
    # is not enough, it needs distribution metadata. A minimal
    # dist-info lives in the project tree at models/_shims/ (NOT in
    # the venv): putting that dir on sys.path makes the version lookup
    # return "0.0.0" through stock importlib.metadata behaviour.
    shims_dir = str(_PROJECT_ROOT / "models" / "_shims")
    if shims_dir not in sys.path:
        sys.path.insert(0, shims_dir)


def _musicgen_weights_present() -> bool:
    """True when the local weights dir holds actual model weight files.

    A lone ``config.json`` does NOT count — a half-finished download
    (configs land before the multi-GB safetensors shards) must not make
    the server claim the model is ready.
    """
    if not MUSICGEN_WEIGHTS_DIR.is_dir():
        return False
    return bool(list(MUSICGEN_WEIGHTS_DIR.glob("*.safetensors")))


def _musicgen_available() -> tuple[bool, str]:
    """Check whether the real model can run in this process.

    Returns ``(available, reason)``. The model needs torch +
    transformers importable AND the weights on disk. Any missing piece
    is a clean "not available" with the reason — never an exception.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "torch is not installed in the project venv"
    _ensure_torchvision_compat()
    _ensure_torchaudio_stub()
    try:
        import transformers  # noqa: F401
        from transformers import (  # noqa: F401
            MusicgenMelodyForConditionalGeneration,
            MusicgenMelodyProcessor,
        )
    except ImportError as exc:
        return False, f"transformers MusicGen-Melody classes unavailable: {exc}"
    if not _musicgen_weights_present():
        return (
            False,
            f"MusicGen-Melody weights not found under {MUSICGEN_WEIGHTS_DIR}",
        )
    return True, "musicgen-melody ready"


def _load_musicgen() -> tuple[Any, Any]:
    """Load (or reuse the cached) MusicGen-Melody model + processor.

    The model is cached in-process: at most one heavy model is ever
    resident (see module docstring). Loading runs inside
    ``model_session("MusicGen-Melody")`` for the VRAM discipline.
    On CPU the weights load in bfloat16 (~3 GiB); on CUDA in float16.
    """
    if "model" in _MODEL_CACHE:
        return _MODEL_CACHE["model"], _MODEL_CACHE["processor"]
    # The media shims must run before ANY transformers import in
    # this process, even when _load_musicgen is called directly without
    # going through _musicgen_available() first.
    _ensure_torchvision_compat()
    _ensure_torchaudio_stub()
    import torch
    from transformers import (
        MusicgenMelodyForConditionalGeneration,
        MusicgenMelodyProcessor,
    )

    device = get_device()
    dtype = torch.bfloat16 if device == "cpu" else torch.float16
    with model_session("MusicGen-Melody"):
        processor = MusicgenMelodyProcessor.from_pretrained(
            str(MUSICGEN_WEIGHTS_DIR)
        )
        model = MusicgenMelodyForConditionalGeneration.from_pretrained(
            str(MUSICGEN_WEIGHTS_DIR), dtype=dtype
        )
        model.to(device)
        model.eval()
    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["processor"] = processor
    _MODEL_CACHE["device"] = device
    return model, processor


def _resample_to_studio(
    audio_32k: np.ndarray, sr_in: int = MUSICGEN_SR
) -> np.ndarray:
    """Resample model output (32 kHz) to the 44 100 Hz studio rate.

    Exact rational resampling via ``scipy.signal.resample_poly`` with
    44100/32000 = 441/320, per channel. Output is float64 stereo.
    """
    from scipy.signal import resample_poly

    a = np.asarray(audio_32k, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, np.newaxis]
    channels = [
        resample_poly(a[:, ch], 441, 320) for ch in range(a.shape[1])
    ]
    return np.column_stack(channels)


def _conditioning_audio_32k(audio_44k: np.ndarray) -> np.ndarray:
    """Down-mix + resample a 44.1 kHz stereo conditioning clip to 32 kHz
    mono — the format the MusicGen-Melody processor expects."""
    from scipy.signal import resample_poly

    a = np.asarray(audio_44k, dtype=np.float64)
    mono = a.mean(axis=1) if a.ndim == 2 else a
    return resample_poly(mono, 320, 441)


def _generate_with_musicgen(
    prompt: str,
    duration_s: float,
    conditioning_44k: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Generate ``duration_s`` seconds with MusicGen-Melody.

    ``conditioning_44k`` is a DSP clip (kick pattern / sweep / scale
    melody) the model follows; the prompt carries the style. Output is
    resampled to 44 100 Hz and trimmed/padded to the exact sample count.
    Raises on any model failure so the caller can fall back honestly.
    """
    import torch

    model, processor = _load_musicgen()
    device = _MODEL_CACHE.get("device", get_device())
    max_new_tokens = max(1, int(round(duration_s * MUSICGEN_FRAME_RATE)))
    melody_32k = _conditioning_audio_32k(conditioning_44k)
    inputs = processor(
        text=[prompt],
        audio=melody_32k,
        sampling_rate=MUSICGEN_SR,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with model_session("MusicGen-Melody"):
        with torch.no_grad():
            # Autocast: the model weights are bf16 (CPU) / fp16 (CUDA)
            # but the processor's chroma features and the T5 text
            # encoder emit float32 tensors; without autocast the first
            # Linear hit by a float32 activation dies with "mat1 and
            # mat2 must have the same dtype, Float vs BFloat16".
            # Autocast casts matmul/linear inputs to the model dtype.
            with torch.autocast(
                device_type=device,
                dtype=torch.bfloat16 if device == "cpu" else torch.float16,
            ):
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    guidance_scale=MUSICGEN_GUIDANCE_SCALE,
                )
    # Shape: (batch=1, channels, samples) at 32 kHz. The model output
    # is bf16 on CPU — .numpy() has no bf16 dtype, so cast to float32
    # first (the values are audio samples; the cast is exact enough).
    audio_32k = generated[0].detach().cpu().float().numpy().astype(np.float64)
    audio_32k = np.ascontiguousarray(audio_32k.T)  # (samples, channels)
    audio_44k = _resample_to_studio(audio_32k, MUSICGEN_SR)
    # MusicGen-Melody renders MONO; the studio contract is stereo, so
    # up-mix identical channels (same as the DSP path's _stereo()).
    if audio_44k.shape[1] == 1:
        audio_44k = np.column_stack([audio_44k[:, 0], audio_44k[:, 0]])
    n_samples = _samples_for_seconds(duration_s)
    return _exact_length(audio_44k, n_samples)


def _resolve_engine() -> tuple[str, str]:
    """Pick the engine for this call: real model when it can run, else
    the honest DSP fallback. Returns ``(engine, note)``."""
    available, reason = _musicgen_available()
    if available:
        return "musicgen-melody", reason
    return "dsp-fallback", reason


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------
@mcp.tool()
def generate_beat(
    bpm: float,
    bars: int,
    style: str = "electronic",
    output_path: str | None = None,
    key: str | None = None,
) -> dict[str, Any]:
    """Generate a tempo-locked drum beat WAV, exactly ``bars`` bars long.

    A 4-bar 120 BPM beat is exactly 8.000 s (352800 samples) at 44 100 Hz
    stereo 24-bit. ``style`` is a free-text hint ("tamil kuthu",
    "phonk", "lo-fi", "trap"); unknown styles fall back to a
    four-on-the-floor electronic pattern and say so. ``key`` (the
    Producer's call convention) is accepted and used in the MusicGen
    prompt; it does not change the DSP pattern. ``output_path`` defaults
    to ``outputs/generative/`` when omitted (the Producer passes none).

    Engine: real MusicGen-Melody when it can run, else the deterministic
    DSP synthesis — the result always reports which engine actually
    produced the audio in ``"engine"``.

    Args:
        bpm: Tempo, 40-240 BPM.
        bars: Number of 4/4 bars (>= 1).
        style: Free-text style hint.
        output_path: Destination WAV (24-bit). Optional.
        key: Musical key hint, e.g. "D major" (Producer convention).

    Returns:
        Output path, engine, style resolved, exact length, and audio info.
    """
    bpm_f = _validate_bpm(bpm)
    bars_i = _validate_bars(bars)
    style_s = _validate_style(style)
    out = _validate_output_path(output_path) or _default_output_path("beat")
    n_samples = _samples_for_bars(bpm_f, bars_i)
    duration_s = n_samples / SAMPLE_RATE

    engine, engine_note = _resolve_engine()
    warnings: list[str] = []
    audio: np.ndarray
    resolved_style = style_s
    if engine == "musicgen-melody":
        try:
            # DSP kick pattern as the melody/rhythm condition: the model
            # follows its tempo while generating real drum content.
            cond, _ = _synthesize_beat_dsp(bpm_f, bars_i, style_s)
            prompt = f"{style_s} drum beat"
            if key:
                prompt += f" in {key}"
            prompt += f", {bpm_f:g} BPM, energetic drums, kick and snare"
            audio = _generate_with_musicgen(prompt, duration_s, cond)
            audio = _peak_normalize(audio, BEAT_PEAK)
        except Exception as exc:  # noqa: BLE001 — model failure -> honest DSP
            warnings.append(f"MusicGen generation failed ({exc}); used DSP fallback.")
            engine = "dsp-fallback"
            audio, resolved_style = _synthesize_beat_dsp(bpm_f, bars_i, style_s)
    else:
        warnings.append(f"MusicGen unavailable ({engine_note}); used DSP fallback.")
        audio, resolved_style = _synthesize_beat_dsp(bpm_f, bars_i, style_s)

    audio = _exact_length(np.asarray(audio, dtype=np.float64), n_samples)
    written = _write_wav(out, audio, SAMPLE_RATE)
    result: dict[str, Any] = {
        "output_path": written,
        "engine": engine,
        "engine_note": engine_note,
        "bpm": bpm_f,
        "bars": bars_i,
        "style": style_s,
        "resolved_style": resolved_style,
        "key": key,
        "warnings": warnings,
        **_audio_info(audio, SAMPLE_RATE),
    }
    return _round_result(result)


@mcp.tool()
def generate_riser(
    duration_ms: float | None = None,
    duration_s: float | None = None,
    style: str = "electronic",
    output_path: str | None = None,
) -> dict[str, Any]:
    """Generate a riser/sweep WAV of an exact length.

    Either ``duration_ms`` (the Producer's call convention) or
    ``duration_s`` must be given; ``duration_ms`` wins when both are
    present. Output is 44 100 Hz stereo 24-bit at the exact sample
    count. ``style`` is a free-text hint carried into the MusicGen
    prompt; the DSP riser is a filtered-noise sweep + tonal lift.

    Args:
        duration_ms: Length in milliseconds (> 0). Producer convention.
        duration_s: Length in seconds (> 0). Task convention.
        style: Free-text style hint.
        output_path: Destination WAV (24-bit). Optional.

    Returns:
        Output path, engine, exact length, and audio info.
    """
    if duration_ms is not None:
        try:
            ms = float(duration_ms)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"duration_ms must be a number, got {duration_ms!r}") from exc
        if not np.isfinite(ms) or ms <= 0:
            raise ValueError(f"duration_ms must be > 0, got {duration_ms}")
        seconds = ms / 1000.0
    elif duration_s is not None:
        try:
            seconds = float(duration_s)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"duration_s must be a number, got {duration_s!r}") from exc
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"duration_s must be > 0, got {duration_s}")
    else:
        raise ValueError("generate_riser needs duration_ms or duration_s")
    style_s = _validate_style(style)
    out = _validate_output_path(output_path) or _default_output_path("riser")
    n_samples = _samples_for_seconds(seconds)

    engine, engine_note = _resolve_engine()
    warnings: list[str] = []
    audio: np.ndarray
    if engine == "musicgen-melody":
        try:
            # Rising sine sweep as the melody condition for the model.
            sweep_n = _samples_for_seconds(seconds)
            t = np.arange(sweep_n, dtype=np.float64) / SAMPLE_RATE
            freq = 200.0 * (1200.0 / 200.0) ** (t / seconds)
            phase = 2.0 * np.pi * np.cumsum(freq) / SAMPLE_RATE
            cond = _stereo(np.sin(phase) * np.linspace(0.0, 1.0, sweep_n) ** 2)
            prompt = (
                f"{style_s} rising tension sweep riser, building energy, "
                f"whoosh and noise swell"
            )
            audio = _generate_with_musicgen(prompt, seconds, cond)
            audio = _peak_normalize(audio, RISER_PEAK)
            # Rise gate: the model is conditioned on a rising sweep,
            # not constrained by it — one run produced flat energy
            # (last/first chunk RMS 0.83x) instead of a riser. A loop
            # below the bar is discarded for the exact DSP riser, so a
            # delivered riser always rises; the engine label says which
            # one it is. The gate measures the model audio BEFORE the
            # final _exact_length (the model output is already exact).
            model_ratio = _rise_ratio(audio)
            if model_ratio < RISER_MIN_RISE_RATIO:
                warnings.append(
                    f"MusicGen riser measured rise ratio {model_ratio:.2f}x "
                    f"(bar {RISER_MIN_RISE_RATIO}x); used DSP fallback."
                )
                engine = "dsp-fallback"
                audio = _synthesize_riser_dsp(seconds, style_s)
        except Exception as exc:  # noqa: BLE001 — model failure -> honest DSP
            warnings.append(f"MusicGen generation failed ({exc}); used DSP fallback.")
            engine = "dsp-fallback"
            audio = _synthesize_riser_dsp(seconds, style_s)
    else:
        warnings.append(f"MusicGen unavailable ({engine_note}); used DSP fallback.")
        audio = _synthesize_riser_dsp(seconds, style_s)

    audio = _exact_length(np.asarray(audio, dtype=np.float64), n_samples)
    written = _write_wav(out, audio, SAMPLE_RATE)
    result: dict[str, Any] = {
        "output_path": written,
        "engine": engine,
        "engine_note": engine_note,
        "duration_ms": round(seconds * 1000.0, 3),
        "duration_seconds": round(seconds, 3),
        "style": style_s,
        "warnings": warnings,
        # Rise ratio of the DELIVERED audio (last/first chunk RMS, 8
        # chunks); the gate above guarantees >= RISER_MIN_RISE_RATIO
        # for a kept model riser, ~400x for DSP.
        "rise_ratio": _rise_ratio(audio),
        **_audio_info(audio, SAMPLE_RATE),
    }
    return _round_result(result)


@mcp.tool()
def generate_melody(
    bpm: float,
    bars: int,
    scale: str = "C major",
    output_path: str | None = None,
    style: str = "electronic",
) -> dict[str, Any]:
    """Generate a melodic loop WAV in the given scale, exactly ``bars``
    bars long at 44 100 Hz stereo 24-bit.

    ``scale`` is e.g. "D major", "A minor", "C major pentatonic". The DSP
    melody is a deterministic random-walk pluck line quantized to the
    scale (every note in-scale by construction) with a root bass note on
    each bar start; the MusicGen path conditions on that same DSP melody
    and elaborates it.

    Args:
        bpm: Tempo, 40-240 BPM.
        bars: Number of 4/4 bars (>= 1).
        scale: Scale name, e.g. "D major".
        output_path: Destination WAV (24-bit). Optional.
        style: Free-text style hint.

    Returns:
        Output path, engine, scale, exact length, and audio info.
    """
    bpm_f = _validate_bpm(bpm)
    bars_i = _validate_bars(bars)
    scale_s = _validate_style(scale)
    style_s = _validate_style(style)
    out = _validate_output_path(output_path) or _default_output_path("melody")
    n_samples = _samples_for_bars(bpm_f, bars_i)
    duration_s = n_samples / SAMPLE_RATE

    engine, engine_note = _resolve_engine()
    warnings: list[str] = []
    # The DSP melody is the condition for the model AND the fallback.
    dsp_audio, canonical_scale = _synthesize_melody_dsp(
        bpm_f, bars_i, scale_s, style_s
    )
    audio: np.ndarray = dsp_audio
    root_midi, intervals, _ = _parse_scale(scale_s)
    pitch_classes = _scale_pitch_classes(root_midi, intervals)
    if engine == "musicgen-melody":
        try:
            prompt = (
                f"{style_s} melodic loop in {canonical_scale}, "
                f"{bpm_f:g} BPM, bright pluck melody"
            )
            audio = _generate_with_musicgen(prompt, duration_s, dsp_audio)
            audio = _peak_normalize(audio, MELODY_PEAK)
            # Scale gate: the model is conditioned on the scale, not
            # constrained by it — three independent runs measured
            # 0.595/0.626/0.758 chroma, i.e. sometimes at chance. A
            # loop below the bar is discarded in favour of the exact
            # DSP fallback (which reads 0.756), so a delivered melody
            # is always in-scale; the engine label says which one it
            # is. The gate measures the model audio BEFORE the final
            # _exact_length (the model output is already exact-length).
            model_chroma = _chroma_in_scale_fraction(audio, pitch_classes)
            if model_chroma < MELODY_MIN_CHROMA:
                warnings.append(
                    f"MusicGen loop measured chroma {model_chroma:.3f} in "
                    f"{canonical_scale} (bar {MELODY_MIN_CHROMA}); used DSP "
                    "fallback."
                )
                engine = "dsp-fallback"
                audio = dsp_audio
        except Exception as exc:  # noqa: BLE001 — model failure -> honest DSP
            warnings.append(f"MusicGen generation failed ({exc}); used DSP fallback.")
            engine = "dsp-fallback"
            audio = dsp_audio
    else:
        warnings.append(f"MusicGen unavailable ({engine_note}); used DSP fallback.")

    audio = _exact_length(np.asarray(audio, dtype=np.float64), n_samples)
    written = _write_wav(out, audio, SAMPLE_RATE)
    # Measured scale adherence of the DELIVERED audio (librosa chroma,
    # 0-1; chance 0.583 for a 7-note scale). Reported for the
    # Producer/Critic to judge; the gate above guarantees it is
    # >= MELODY_MIN_CHROMA for a kept model loop, ~0.756 for DSP.
    scale_adherence = _chroma_in_scale_fraction(audio, pitch_classes)
    result: dict[str, Any] = {
        "output_path": written,
        "engine": engine,
        "engine_note": engine_note,
        "bpm": bpm_f,
        "bars": bars_i,
        "scale": canonical_scale,
        "style": style_s,
        "warnings": warnings,
        "scale_adherence_chroma": scale_adherence,
        **_audio_info(audio, SAMPLE_RATE),
    }
    return _round_result(result)


@mcp.tool()
def engine_status() -> dict[str, Any]:
    """Report which generative engine is active and why.

    No audio is generated. Returns the resolved engine
    ("musicgen-melody" or "dsp-fallback"), the reason, device, the
    registered VRAM estimate for MusicGen-Melody, and whether the
    weights are present on disk.
    """
    available, reason = _musicgen_available()
    return _round_result(
        {
            "engine": "musicgen-melody" if available else "dsp-fallback",
            "reason": reason,
            "device": get_device(),
            "model_id": MUSICGEN_MODEL_ID,
            "weights_dir": str(MUSICGEN_WEIGHTS_DIR),
            "weights_present": _musicgen_weights_present(),
            "vram_estimate_gb": MODEL_VRAM_GB.get("MusicGen-Melody"),
            "model_loaded_in_process": "model" in _MODEL_CACHE,
            "sample_rate": SAMPLE_RATE,
        }
    )


@mcp.tool()
def unload_model() -> dict[str, Any]:
    """Free the cached MusicGen model (if loaded) and empty the CUDA cache.

    The server keeps at most one model resident; this tool releases it
    explicitly, e.g. before another server loads its own heavy model.
    Returns whether a model was actually unloaded.
    """
    unloaded = "model" in _MODEL_CACHE
    _MODEL_CACHE.clear()
    gc.collect()
    if get_device() == "cuda":
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — cache emptying is best-effort
            pass
    return {"unloaded": unloaded, "model_loaded_in_process": False}


def main() -> None:
    """Run the generative audio MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
