"""Stem + Theory MCP Server — Autonomous Multi-Agent AI Music Producer & Remix Studio.

Exposes real neural stem separation plus harmony / raga analysis as Model
Context Protocol tools so the LangGraph Producer agent can drive them over
file paths — never raw audio arrays over MCP.

Transport: stdio (``python mcp_servers/stem_theory_server.py``).
Project standard sample rate: 44_100 Hz. All file I/O goes through
``soundfile``; outputs are written as 24-bit WAV (``subtype='PCM_24'``)
with exactly the input length in samples.

Separation model
---------------
BS-RoFormer, 4-stem checkpoint ``bs_roformer_4stems_ft`` (Apache-2.0,
Hugging Face ``SYH99999/bs_roformer_4stems_ft``), run through the
``audio-separator`` library (``MDXCSeparator`` instantiated directly —
the checkpoint is not in audio-separator's built-in UVR model list, so
the stock ``Separator.load_model`` path cannot find it). The model emits
four stems — ``drums``, ``bass``, ``other``, ``vocals`` — in that order.

Five stems out, honestly labelled
---------------------------------
The Producer contract wants five stems: vocals, backing, drums, bass,
melodic. The model natively produces four, so the fifth split is derived
from the model's ``other`` stem with librosa HPSS (harmonic / percussive
separation) and labelled as such in every result — never presented as
model output:

* ``vocals``   — BS-RoFormer vocals stem.
* ``drums``    — BS-RoFormer drums stem.
* ``bass``     — BS-RoFormer bass stem.
* ``melodic``  — HPSS *harmonic* part of the BS-RoFormer ``other`` stem
  (pitched instruments: keys, guitar, synth).
* ``backing``  — HPSS *percussive* part of the BS-RoFormer ``other`` stem
  (rhythmic texture, backing-vocal bleed, strums). ``melodic + backing``
  reconstructs ``other`` sample-exactly (HPSS partitions the spectrum).

VRAM discipline
---------------
The model (527 MB weights, ~4 GiB working set per ``core/config.py``)
is loaded inside ``core.config.model_session("BS-RoFormer")`` and
unloaded afterwards (``del`` + ``gc.collect()`` + ``torch.cuda.empty_cache()``
on CUDA), so the server process never holds the model between calls and
the next model in the chain starts from a clean slate. On this CPU-only
machine the session is a logged no-op and inference is slower — correct,
not fast. ``ensure_fits("BS-RoFormer")`` is checked before every load.

No silent fallbacks: if the model file is missing, the checkpoint fails
to load, or inference raises, the tool raises a clear error naming why.
HPSS is used ONLY for the documented melodic/backing split above, never
as a stand-in for the model.
"""

from __future__ import annotations

import gc
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Import shim: make `core.config` importable whether this file is run as a
# script (`python mcp_servers/stem_theory_server.py`, sys.path[0] is
# mcp_servers/) or as a module from the project root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.config import ensure_fits, model_session  # noqa: E402

mcp = FastMCP("stem-theory")

logger = logging.getLogger("stem-theory")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 44100
MIN_DURATION_S = 1.0

#: Model files live under the project dir so a fresh checkout can re-fetch
#: them; never in /tmp (512 MB tmpfs) and never in the user cache.
MODEL_DIR = _PROJECT_ROOT / "models" / "bs_roformer_4stems_ft"
MODEL_FILE = MODEL_DIR / "bs_roformer_4stems_ft.pth"
MODEL_CONFIG_FILE = MODEL_DIR / "config.yaml"
MODEL_NAME = "bs_roformer_4stems_ft"
MODEL_SOURCE = "SYH99999/bs_roformer_4stems_ft (Hugging Face, Apache-2.0)"

#: Stems the BS-RoFormer checkpoint natively produces (config.yaml
#: training.instruments order).
MODEL_STEMS = ("drums", "bass", "other", "vocals")

#: Stems this server returns to the Producer, in contract order.
OUTPUT_STEMS = ("vocals", "backing", "drums", "bass", "melodic")

#: audio-separator normalisation thresholds chosen to PRESERVE the model's
#: natural stem levels: stems are only scaled down if they exceed full
#: scale, never scaled up (amplification_threshold 0.0 disables the
#: up-scaling branch). The stock 0.9 default would re-peak every stem
#: and destroy their mix-relative balance.
NORMALIZATION_THRESHOLD = 1.0
AMPLIFICATION_THRESHOLD = 0.0

#: CPU inference batch size. The model YAML asks for 2, but this box has
#: 8 GiB RAM and the fp32 activations of one ~11 s chunk are already
#: large; batch 1 halves peak memory at a small speed cost.
INFERENCE_BATCH_SIZE = 1

#: Krumhansl-Schmuckler key profiles (major / minor), C-rooted, in
#: pitch-class order C C# D D# E F F# G G# A A# B.
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
_PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

#: Chord templates (major / minor triads), pitch-class masks.
_CHORD_TEMPLATES: dict[str, tuple[int, int, int]] = {
    **{f"{name}": (i, (i + 4) % 12, (i + 7) % 12) for i, name in enumerate(_PITCH_CLASS_NAMES)},
    **{f"{name}m": (i, (i + 3) % 12, (i + 7) % 12) for i, name in enumerate(_PITCH_CLASS_NAMES)},
}

#: Minimum template correlation to name a chord; below this the window is
#: reported as "N.C." (no chord) rather than a guessed chord.
CHORD_MIN_CORRELATION = 0.5
#: Minimum key-profile correlation to name a key; below this the key is
#: reported as "unknown".
KEY_MIN_CORRELATION = 0.5

#: Analysis window for the chord sequence (seconds).
CHORD_WINDOW_S = 2.0

#: Small raga scale dictionary: raga -> pitch-class set (relative to the
#: tonic, 0 = Sa). Only ragas whose scale is unambiguous from a
#: pitch-class profile are listed; anything else returns "unknown".
_RAGA_SCALES: dict[str, frozenset[int]] = {
    "Mohanam": frozenset({0, 2, 4, 7, 9}),
    "Hamsadhwani": frozenset({0, 2, 4, 7, 11}),
    "Kalyani": frozenset({0, 2, 4, 6, 7, 9, 11}),
    "Shankarabharanam": frozenset({0, 2, 4, 5, 7, 9, 11}),
    "Kharaharapriya": frozenset({0, 2, 3, 5, 7, 8, 10}),
    "Natabhairavi": frozenset({0, 2, 3, 5, 7, 8, 10}),
    "Mayamalavagowla": frozenset({0, 1, 4, 5, 7, 8, 11}),
    "Hindolam": frozenset({0, 3, 5, 7, 10}),
    "Abheri": frozenset({0, 2, 3, 5, 7, 9, 10}),
}
#: Minimum fraction of chroma energy on scale tones to name a raga.
RAGA_MIN_SCALE_ENERGY = 0.85


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _f(value: Any, default: float = 0.0) -> float:
    """Coerce a possibly-numpy scalar to a plain JSON-safe float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round(value: float, ndigits: int = 3) -> float:
    """Round a float for JSON output (NaN/inf become 0.0, never leak)."""
    value = _f(value)
    if not np.isfinite(value):
        return 0.0
    return round(value, ndigits)


def _resolve_input(audio_path: str = "", input_path: str = "") -> Path:
    """Accept the Producer's ``audio_path`` and the task's ``input_path``
    spellings; exactly one must name an existing file."""
    chosen = (audio_path or "").strip() or (input_path or "").strip()
    if not chosen:
        raise ValueError("no input audio path given (audio_path / input_path)")
    path = Path(chosen)
    if not path.is_file():
        raise FileNotFoundError(f"input audio not found: {path}")
    return path


def _read_wav_validated(path: Path) -> tuple[np.ndarray, int, int]:
    """Read a WAV as ``(samples, channels)`` float64 plus its sample rate
    and sample count. Validates the studio contract: 44 100 Hz, >= 1 s.
    Raises a clear ``ValueError`` otherwise — never resamples silently."""
    try:
        info = sf.info(str(path))
    except Exception as exc:
        raise ValueError(f"cannot read audio file {path}: {exc}") from exc
    if info.samplerate != SAMPLE_RATE:
        raise ValueError(
            f"expected {SAMPLE_RATE} Hz studio-wide, got {info.samplerate} Hz: {path}"
        )
    if info.duration < MIN_DURATION_S:
        raise ValueError(
            f"input too short: {info.duration:.2f} s < {MIN_DURATION_S:.0f} s minimum: {path}"
        )
    audio, sr = sf.read(str(path), always_2d=True, dtype="float64")
    if audio.shape[0] == 0 or not np.any(audio):
        raise ValueError(f"input audio is empty or all silence: {path}")
    return np.asarray(audio, dtype=np.float64), int(sr), int(audio.shape[0])


def _write_wav(path: str | Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    """Write ``(samples, channels)`` float audio as 24-bit WAV; return str path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float64), sr, subtype="PCM_24")
    return str(path)


def _fit_length(audio: np.ndarray, n_samples: int) -> np.ndarray:
    """Trim or zero-pad a stem to exactly ``n_samples`` rows. Model
    chunking can leave a stem a few samples long/short; the contract is
    exact input length, so this is enforced here, never assumed."""
    n = audio.shape[0]
    if n == n_samples:
        return audio
    if n > n_samples:
        return np.ascontiguousarray(audio[:n_samples])
    pad = np.zeros((n_samples - n, audio.shape[1]), dtype=np.float64)
    return np.concatenate([audio, pad], axis=0)


def _audio_info(audio: np.ndarray, sr: int) -> dict[str, Any]:
    """Basic shape/duration facts shared by several tools."""
    channels = 1 if audio.ndim == 1 else int(audio.shape[1])
    num_samples = int(audio.shape[0])
    return {
        "sample_rate": int(sr),
        "channels": channels,
        "num_samples": num_samples,
        "duration_seconds": round(num_samples / sr, 3),
        "duration_ms": int(round(num_samples / sr * 1000.0)),
    }


def _rms_dbfs(audio: np.ndarray) -> float:
    """RMS level in dBFS of a (samples, channels) array (-120 for silence)."""
    if audio.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2)))
    return round(20.0 * np.log10(rms), 3) if rms > 0 else -120.0


# ---------------------------------------------------------------------------
# Model loading (BS-RoFormer via audio-separator, direct instantiation)
# ---------------------------------------------------------------------------
def _load_model_data() -> dict[str, Any]:
    """Parse the checkpoint's config.yaml (PyYAML FullLoader — the file
    uses ``!!python/tuple`` tags) and flag it as a RoFormer model so
    audio-separator routes it to the RoformerLoader."""
    if not MODEL_CONFIG_FILE.is_file():
        raise FileNotFoundError(
            f"BS-RoFormer config not found: {MODEL_CONFIG_FILE} — "
            "download it from Hugging Face SYH99999/bs_roformer_4stems_ft"
        )
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to read the BS-RoFormer config "
            "(pip install pyyaml)"
        ) from exc
    with open(MODEL_CONFIG_FILE, encoding="utf-8") as handle:
        model_data = yaml.load(handle, Loader=yaml.FullLoader)
    if not isinstance(model_data, dict) or "model" not in model_data:
        raise ValueError(
            f"BS-RoFormer config is not a model config: {MODEL_CONFIG_FILE}"
        )
    model_data = dict(model_data)
    model_data["is_roformer"] = True
    return model_data


def _build_separator(output_dir: Path, model_data: dict[str, Any]):
    """Instantiate audio-separator's MDXCSeparator directly against the
    local 4-stem checkpoint.

    The stock ``Separator.load_model`` path cannot be used: it only knows
    the UVR download list, and this checkpoint is not in it (the list
    carries only 2-stem vocal RoFormer models). Direct instantiation
    with the YAML model data is the supported route for custom models.
    """
    import torch
    from audio_separator.separator.architectures.mdxc_separator import (
        MDXCSeparator,
    )

    common_config = {
        "logger": logger,
        "log_level": logging.WARNING,
        "torch_device": torch.device("cpu"),
        "torch_device_cpu": torch.device("cpu"),
        "torch_device_mps": None,
        "onnx_execution_provider": None,
        "model_name": MODEL_NAME,
        "model_path": str(MODEL_FILE),
        "model_data": model_data,
        "output_format": "WAV",
        "output_bitrate": None,
        "output_dir": str(output_dir),
        # Preserve the model's natural stem levels (see the constant's
        # comment): only scale down above full scale, never scale up.
        "normalization_threshold": NORMALIZATION_THRESHOLD,
        "amplification_threshold": AMPLIFICATION_THRESHOLD,
        "output_single_stem": None,
        "invert_using_spec": False,
        "sample_rate": SAMPLE_RATE,
        # soundfile keeps the float path (pydub round-trips through int16
        # and would cap the stems at 16-bit precision).
        "use_soundfile": True,
        "use_autocast": False,
        "use_native_fp16": False,
        "use_torch_compile": False,
    }
    arch_config = {
        "segment_size": 256,
        "override_model_segment_size": False,
        "batch_size": INFERENCE_BATCH_SIZE,
        "overlap": None,  # from the YAML inference.num_overlap (4)
        "pitch_shift": 0,
        "process_all_stems": True,
    }
    try:
        return MDXCSeparator(common_config=common_config, arch_config=arch_config)
    except SystemExit as exc:
        # audio-separator's loader calls sys.exit(1) on a load failure;
        # convert it to a raisable error so the MCP server process lives.
        raise RuntimeError(
            f"BS-RoFormer model failed to load from {MODEL_FILE} "
            f"(audio-separator exited with code {exc.code})"
        ) from exc


def _separate_with_model(
    input_path: Path, work_dir: Path, model_data: dict[str, Any]
) -> dict[str, Path]:
    """Run the model once and return ``{stem_name: raw_output_path}``.

    The separator writes ``<base>_(<stem>)_<model>.wav`` files into
    ``work_dir``; this maps them back to the four model stem names.
    Raises ``RuntimeError`` naming the failure when the model produces
    no usable stems — never fabricates a stem.
    """
    separator = None
    try:
        separator = _build_separator(work_dir, model_data)
        raw_paths = separator.separate(str(input_path))
    finally:
        # Unload the model even on failure: the server process must not
        # hold 527 MB of weights between calls (VRAM discipline).
        if separator is not None:
            try:
                del separator
            except Exception:  # noqa: BLE001 — unload is best-effort
                pass
            gc.collect()
    if not raw_paths:
        raise RuntimeError(
            f"BS-RoFormer separation produced no output files for {input_path}"
        )
    stems: dict[str, Path] = {}
    for raw in raw_paths:
        # audio-separator returns BARE filenames (get_stem_output_path);
        # the files actually live in work_dir (final_process joins the
        # output dir when writing, but returns the relative name).
        raw_path = work_dir / Path(raw).name
        if not raw_path.is_file():
            raise RuntimeError(
                f"BS-RoFormer separation reported {raw!r} but no file "
                f"exists at {raw_path}"
            )
        name = raw_path.name
        for stem in MODEL_STEMS:
            if f"_({stem})_" in name:
                stems[stem] = raw_path
                break
    missing = [s for s in MODEL_STEMS if s not in stems]
    if missing:
        raise RuntimeError(
            f"BS-RoFormer separation is missing stems {missing} "
            f"(got files: {[Path(p).name for p in raw_paths]})"
        )
    return stems


def _split_other(other: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Split the BS-RoFormer ``other`` stem into melodic / backing with
    librosa HPSS (harmonic / percussive). This is the ONLY HPSS in the
    separation path and it is labelled as a derivation, never as model
    output: the model has no 'melodic' or 'backing' target. HPSS
    partitions the spectrum, so melodic + backing reconstructs ``other``
    sample-exactly (up to float rounding)."""
    import librosa

    harmonic_ch: list[np.ndarray] = []
    percussive_ch: list[np.ndarray] = []
    for ch in range(other.shape[1]):
        harmonic, percussive = librosa.effects.hpss(other[:, ch])
        harmonic_ch.append(np.asarray(harmonic, dtype=np.float64))
        percussive_ch.append(np.asarray(percussive, dtype=np.float64))
    melodic = np.column_stack(harmonic_ch)
    backing = np.column_stack(percussive_ch)
    return melodic, backing


# ---------------------------------------------------------------------------
# Harmony analysis (librosa chroma — no invented chords)
# ---------------------------------------------------------------------------
def _chroma(audio: np.ndarray, sr: int) -> np.ndarray:
    """12-row constant-Q chromagram, one column per ~46 ms frame."""
    import librosa

    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    return librosa.feature.chroma_cqt(y=np.asarray(mono, dtype=np.float64), sr=sr)


def _key_from_chroma(chroma: np.ndarray) -> tuple[str, float]:
    """Krumhansl-Schmuckler key estimate: correlate the mean chroma
    vector against all 24 rotated key profiles. Returns (key, r) where
    r is the Pearson correlation; r < KEY_MIN_CORRELATION means the
    caller reports 'unknown'."""
    mean_chroma = chroma.mean(axis=1)
    if float(np.sum(mean_chroma)) <= 0.0:
        return "unknown", 0.0
    mean_chroma = mean_chroma / float(np.sum(mean_chroma))
    best_key = "unknown"
    best_r = -1.0
    for shift in range(12):
        for profile, suffix in ((_MAJOR_PROFILE, ""), (_MINOR_PROFILE, "m")):
            rotated = np.roll(profile, shift)
            rotated = rotated / float(np.sum(rotated))
            denom = float(
                np.std(mean_chroma) * np.std(rotated) * len(mean_chroma)
            )
            if denom <= 0.0:
                continue
            r = float(
                np.sum((mean_chroma - mean_chroma.mean()) * (rotated - rotated.mean()))
                / denom
            )
            if r > best_r:
                best_r = r
                best_key = f"{_PITCH_CLASS_NAMES[shift]}{' minor' if suffix else ' major'}"
    return best_key, best_r


def _chords_from_chroma(
    chroma: np.ndarray, sr: int
) -> list[dict[str, Any]]:
    """One chord per CHORD_WINDOW_S window: correlate the window's mean
    chroma against 24 major/minor triad templates. Windows below
    CHORD_MIN_CORRELATION are reported as 'N.C.' — no chord is ever
    guessed from a weak match."""
    hop_s = 512 / sr  # librosa chroma_cqt default hop_length is 512
    window_frames = max(1, int(round(CHORD_WINDOW_S / hop_s)))
    chords: list[dict[str, Any]] = []
    n_frames = chroma.shape[1]
    for start in range(0, n_frames, window_frames):
        end = min(n_frames, start + window_frames)
        window = chroma[:, start:end].mean(axis=1)
        total = float(np.sum(window))
        start_s = round(start * hop_s, 3)
        end_s = round(end * hop_s, 3)
        if total <= 0.0:
            chords.append(
                {"start_s": start_s, "end_s": end_s, "chord": "N.C.", "confidence": 0.0}
            )
            continue
        window = window / total
        best_chord = "N.C."
        best_r = -1.0
        for name, tones in _CHORD_TEMPLATES.items():
            template = np.zeros(12, dtype=np.float64)
            template[list(tones)] = 1.0
            template = template / 3.0
            denom = float(np.std(window) * np.std(template) * 12)
            if denom <= 0.0:
                continue
            r = float(
                np.sum((window - window.mean()) * (template - template.mean())) / denom
            )
            if r > best_r:
                best_r = r
                best_chord = name
        if best_r < CHORD_MIN_CORRELATION:
            chords.append(
                {"start_s": start_s, "end_s": end_s, "chord": "N.C.", "confidence": _round(max(best_r, 0.0))}
            )
        else:
            chords.append(
                {"start_s": start_s, "end_s": end_s, "chord": best_chord, "confidence": _round(best_r)}
            )
    return chords


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------
@mcp.tool()
def separate_stems(
    audio_path: str = "",
    input_path: str = "",
    model: str = "BS-RoFormer",
    output_dir: str = "",
) -> dict[str, Any]:
    """Separate a mix into 5 stems with the BS-RoFormer neural model.

    The model natively produces drums / bass / other / vocals; ``melodic``
    and ``backing`` are the HPSS harmonic / percussive split of the
    model's ``other`` stem (labelled as derived, never as model output —
    see the module docstring). All stems are 44 100 Hz stereo 24-bit WAV
    at exactly the input length in samples.

    Args:
        audio_path: Input WAV file (the Producer calls it this).
        input_path: Alias for audio_path.
        model: Separation model; only "BS-RoFormer" is supported.
        output_dir: Destination directory for the stems. Defaults to
            ``<input-stem>_stems`` next to the input file.

    Returns:
        Dict with the 5 stem paths, per-stem derivation + RMS level,
        model identity, and input audio info. Raises a clear error when
        the input is invalid or the model cannot run — never fakes stems.
    """
    if model not in ("BS-RoFormer", "bs-roformer", "bs_roformer"):
        raise ValueError(
            f"unsupported separation model: {model!r} — only 'BS-RoFormer' is supported"
        )
    in_path = _resolve_input(audio_path, input_path)
    audio, sr, n_samples = _read_wav_validated(in_path)
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(
            f"BS-RoFormer weights not found: {MODEL_FILE} — "
            "download bs_roformer_4stems_ft.pth from Hugging Face "
            "SYH99999/bs_roformer_4stems_ft into models/bs_roformer_4stems_ft/"
        )
    if not ensure_fits("BS-RoFormer"):
        raise RuntimeError(
            "BS-RoFormer (4.0 GiB est.) does not fit the VRAM budget; "
            "free GPU memory or run on CPU"
        )

    out_dir = Path(output_dir) if output_dir else in_path.parent / f"{in_path.stem}_stems"
    out_dir.mkdir(parents=True, exist_ok=True)
    # audio-separator writes its own files into a scratch dir first; the
    # server then re-writes them as clean-named PCM_24 stems at exact
    # input length. Scratch lives next to the output, never in /tmp.
    work_dir = out_dir / "_separator_raw"
    work_dir.mkdir(parents=True, exist_ok=True)

    model_data = _load_model_data()
    with model_session("BS-RoFormer"):
        raw_stems = _separate_with_model(in_path, work_dir, model_data)

    stems: dict[str, str] = {}
    derivation: dict[str, str] = {}
    rms_dbfs: dict[str, float] = {}
    # 1) The three native model stems, re-written as PCM_24 at exact length.
    for stem in ("vocals", "drums", "bass"):
        raw_audio, _ = sf.read(str(raw_stems[stem]), always_2d=True, dtype="float64")
        fitted = _fit_length(np.asarray(raw_audio, dtype=np.float64), n_samples)
        stems[stem] = _write_wav(out_dir / f"{stem}.wav", fitted, sr)
        derivation[stem] = "bs-roformer"
        rms_dbfs[stem] = _rms_dbfs(fitted)
    # 2) The derived split of the model's "other" stem (HPSS, labelled).
    other_audio, _ = sf.read(str(raw_stems["other"]), always_2d=True, dtype="float64")
    other_audio = _fit_length(np.asarray(other_audio, dtype=np.float64), n_samples)
    melodic, backing = _split_other(other_audio, sr)
    stems["melodic"] = _write_wav(out_dir / "melodic.wav", melodic, sr)
    stems["backing"] = _write_wav(out_dir / "backing.wav", backing, sr)
    derivation["melodic"] = "hpss-harmonic of bs-roformer 'other' stem"
    derivation["backing"] = "hpss-percussive of bs-roformer 'other' stem"
    rms_dbfs["melodic"] = _rms_dbfs(melodic)
    rms_dbfs["backing"] = _rms_dbfs(backing)
    # Keep the model's "other" stem on disk too — it is the source of the
    # derived split and the audit trail for it.
    stems_other = _write_wav(out_dir / "other.wav", other_audio, sr)

    return {
        "stems": {name: stems[name] for name in OUTPUT_STEMS},
        "other_stem": stems_other,
        "derivation": derivation,
        "rms_dbfs": rms_dbfs,
        "model": "BS-RoFormer",
        "model_checkpoint": MODEL_NAME,
        "model_source": MODEL_SOURCE,
        "input_audio": str(in_path),
        **_audio_info(audio, sr),
    }


@mcp.tool()
def analyze_harmony(
    audio_path: str = "",
    input_path: str = "",
) -> dict[str, Any]:
    """Estimate the key and chord sequence of an audio file from its chroma.

    Key: Krumhansl-Schmuckler correlation of the mean chroma vector
    against all 24 major/minor profiles. Chords: one per 2-second window
    by template correlation against major/minor triads. Windows that do
    not match any template are reported as "N.C." — chords are never
    invented from a weak match.

    Args:
        audio_path: Input WAV file.
        input_path: Alias for audio_path.

    Returns:
        Key, key confidence, and the timestamped chord sequence.
    """
    in_path = _resolve_input(audio_path, input_path)
    audio, sr, _ = _read_wav_validated(in_path)
    chroma = _chroma(audio, sr)
    key, key_r = _key_from_chroma(chroma)
    chords = _chords_from_chroma(chroma, sr)
    return {
        "input_audio": str(in_path),
        "key": key if key_r >= KEY_MIN_CORRELATION else "unknown",
        "key_confidence": _round(max(key_r, 0.0)),
        "key_candidate": key,
        "chords": chords,
        **_audio_info(audio, sr),
    }


@mcp.tool()
def detect_raga_tala(
    audio_path: str = "",
    input_path: str = "",
) -> dict[str, Any]:
    """Heuristic raga / tala hints from the chroma + onset pattern.

    Raga: the pitch-class energy profile is matched against a small
    dictionary of Carnatic raga scales; the raga is named only when at
    least 85% of the chroma energy sits on its scale tones, otherwise
    "unknown". Tala: the onset-autocorrelation period is checked against
    common cycles (Adi 8 beats, Rupakam 6, Misra Chapu 7); it is a hint,
    not a measurement, and returns "unknown" unless the period is
    unambiguous. Nothing here is a definitive identification — the
    Director treats these as hints and never as ground truth.

    Args:
        audio_path: Input WAV file.
        input_path: Alias for audio_path.

    Returns:
        Raga + tala hints with confidences, or "unknown" with low
        confidence when the evidence is weak.
    """
    import librosa

    in_path = _resolve_input(audio_path, input_path)
    audio, sr, n_samples = _read_wav_validated(in_path)
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio[:, 0]
    chroma = _chroma(audio, sr)
    mean_chroma = chroma.mean(axis=1)
    total = float(np.sum(mean_chroma))
    if total <= 0.0:
        return {
            "input_audio": str(in_path),
            "raga": "unknown",
            "raga_confidence": 0.0,
            "tala": "unknown",
            "tala_confidence": 0.0,
            "method": "chroma + onset autocorrelation (heuristic)",
            **_audio_info(audio, sr),
        }
    profile = mean_chroma / total

    # --- Raga: best scale match over all 12 tonic rotations. ---
    best_raga = "unknown"
    best_energy = 0.0
    best_tonic = 0
    for tonic in range(12):
        rotated = np.roll(profile, -tonic)
        for raga, scale in _RAGA_SCALES.items():
            on_scale = float(sum(rotated[p] for p in scale))
            if on_scale > best_energy:
                best_energy = on_scale
                best_raga = raga
                best_tonic = tonic
    raga = best_raga if best_energy >= RAGA_MIN_SCALE_ENERGY else "unknown"
    raga_confidence = _round(best_energy if raga != "unknown" else best_energy)

    # --- Tala: onset-autocorrelation period in beats. ---
    tala = "unknown"
    tala_confidence = 0.0
    try:
        tempo, _beats = librosa.beat.beat_track(y=mono, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0]) if tempo is not None else 0.0
        onset_env = librosa.onset.onset_strength(y=mono, sr=sr)
        if bpm > 0 and onset_env.size > 0:
            ac = librosa.autocorrelate(onset_env, max_size=onset_env.size)
            ac[: int(sr / 512)] = 0.0  # ignore sub-beat lags
            # Lags in beats: hop 512 @ 44.1 kHz, beat = 60/bpm seconds.
            frames_per_beat = (60.0 / bpm) * sr / 512.0
            candidates = {
                "Adi (8-beat)": 8.0,
                "Rupakam (6-beat)": 6.0,
                "Misra Chapu (7-beat)": 7.0,
                "Khanda Chapu (5-beat)": 5.0,
            }
            scored: list[tuple[float, str]] = []
            for name, beats in candidates.items():
                lag = int(round(beats * frames_per_beat))
                if 0 < lag < len(ac):
                    peak = float(np.max(ac[max(0, lag - 2) : lag + 3]))
                    scored.append((peak, name))
            if scored:
                scored.sort(reverse=True)
                best_peak, best_tala = scored[0]
                second_peak = scored[1][0] if len(scored) > 1 else 0.0
                # Unambiguous only when the best cycle dominates the runner-up.
                if best_peak > 0 and best_peak >= 1.5 * max(second_peak, 1e-9):
                    tala = best_tala
                    tala_confidence = _round(min(best_peak / (best_peak + second_peak), 1.0))
    except Exception:  # noqa: BLE001 — tala is best-effort, never fatal
        tala = "unknown"
        tala_confidence = 0.0

    return {
        "input_audio": str(in_path),
        "raga": raga,
        "raga_confidence": raga_confidence,
        "raga_tonic": _PITCH_CLASS_NAMES[best_tonic] if raga != "unknown" else None,
        "tala": tala,
        "tala_confidence": tala_confidence,
        "method": "chroma scale match + onset autocorrelation (heuristic)",
        **_audio_info(audio, sr),
    }


def main() -> None:
    """Run the stem-theory MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
