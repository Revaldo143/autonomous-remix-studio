"""DAW Export MCP Server — Autonomous Multi-Agent AI Music Producer & Remix Studio.

Exposes audio-to-MIDI transcription and DAW project export as Model
Context Protocol tools so the LangGraph agents (and the user) can take a
finished remix out of the studio and into a real DAW:

* ``audio_to_midi`` — transcribe an audio file to a valid ``.mid`` file.
  Basic Pitch is the preferred transcriber, but it genuinely cannot run
  in this environment (see ``_basic_pitch_status``): installing it would
  force a rebuild of the pinned ``numpy==2.5.5`` against an old,
  Python-3.12-incompatible source tree, and it pulls a TensorFlow stack
  the pinned environment cannot take. When Basic Pitch is unavailable the
  tool falls back to a clearly-labelled onset-to-MIDI conversion
  (librosa onset detection + piptrack pitch) and says so in every return
  value — a fallback is never presented as Basic Pitch.
* ``export_daw_project`` — write a REAPER project file (``.RPP``, plain
  text) with one track per stem WAV, each clip placed at its correct
  start time, tempo set from the blueprint, plus one MIDI track per
  supplied ``.mid`` file. The referenced stems + MIDI files are copied
  into the canonical ``stems_and_midi/`` layout next to the ``.RPP`` so
  the project is portable (relative ``FILE`` paths only).
* ``export_stems_and_midi`` — copy stems + MIDI files into the canonical
  ``stems_and_midi/`` layout with a ``manifest.json`` (file list, sample
  rate, durations, bpm).

Transport: stdio (``python mcp_servers/daw_export_server.py``).
Project standard sample rate: 44_100 Hz. All times are exact sample
indices converted to seconds; no fake content — a track in the ``.RPP``
always points at a real file that exists.

VRAM hygiene
------------
Basic Pitch is a ~0.5 GiB model (see ``core.config.MODEL_VRAM_GB``).
When it is available it is loaded inside a ``core.config.model_session``
window and unloaded afterwards, so a single 16 GB consumer GPU stays
within budget. The onset fallback is pure NumPy/SciPy/librosa and loads
no model at all.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Import shim: make `agents.state` / `core.config` importable whether this
# file is run as a script (`python mcp_servers/daw_export_server.py`,
# sys.path[0] is mcp_servers/) or as a module from the project root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.state import ArrangementBlueprint, SAMPLE_RATE  # noqa: E402

try:  # noqa: E402 — core.config is CPU-safe (torch is an optional import)
    from core.config import model_session as _model_session
except Exception:  # pragma: no cover - defensive: config must never break export
    _model_session = None  # type: ignore[assignment]

mcp = FastMCP("daw-export")

#: Directory Basic Pitch model weights live under when the model is used.
#: (No weights are bundled here — Basic Pitch is unavailable in this
#: environment; see `_basic_pitch_status`.)
BASIC_PITCH_MODEL_DIR = _PROJECT_ROOT / "models" / "basic-pitch"

#: Stem filenames that are render outputs, not stems — never given a track.
_NON_STEM_NAMES = ("final_master_full.wav",)
_NON_STEM_PREFIXES = ("viral_hook",)

#: Canonical track order in the .RPP (roles not listed sort after these,
#: alphabetically).
_TRACK_ORDER = ("vocal", "drums", "bass", "melodic", "backing", "kick", "fx")

#: Human-readable track names per stem role.
_ROLE_DISPLAY = {
    "vocal": "Vocals",
    "drums": "Drums",
    "bass": "Bass",
    "melodic": "Melodic",
    "backing": "Backing",
    "kick": "Kick Pattern",
    "fx": "FX",
}

# ---------------------------------------------------------------------------
# Basic Pitch availability — checked lazily, cached, never fatal.
# ---------------------------------------------------------------------------
_basic_pitch_status_cache: dict[str, Any] | None = None


def _basic_pitch_status() -> dict[str, Any]:
    """Probe whether Basic Pitch can actually run here (import + TF stack).

    Returns ``{"available": bool, "reason": str}``. The import is attempted
    once and cached. A failed import is a fact about this environment, not
    an error to raise — callers fall back to onset-to-MIDI and report it.
    """
    global _basic_pitch_status_cache
    if _basic_pitch_status_cache is not None:
        return _basic_pitch_status_cache
    try:
        import basic_pitch  # noqa: F401
        from basic_pitch.inference import predict  # noqa: F401

        _basic_pitch_status_cache = {
            "available": True,
            "reason": "basic_pitch importable",
        }
    except Exception as exc:  # noqa: BLE001 — any import failure means unavailable
        _basic_pitch_status_cache = {
            "available": False,
            "reason": (
                "Basic Pitch is not installed in this environment: "
                f"{type(exc).__name__}: {exc}. "
                "Installing it would force a rebuild of the pinned "
                "numpy==2.5.5 against an old, Python-3.12-incompatible "
                "source tree (pip dry-run fails building 'numpy'), and it "
                "pulls a TensorFlow stack the pinned environment cannot "
                "take. Onset-to-MIDI fallback is used instead."
            ),
        }
    return _basic_pitch_status_cache


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _f(value: Any, default: float = 0.0) -> float:
    """Coerce a possibly-numpy scalar to a plain JSON-safe float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _new_guid() -> str:
    """A REAPER-style GUID, e.g. ``{3F2504E0-4F89-41D3-9A0C-0305E82C330D}``."""
    return "{" + str(uuid.uuid4()).upper() + "}"


def _rpp_name(name: str) -> str:
    """Escape a track/item name for a quoted REAPER string."""
    return '"' + name.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _fmt_seconds(seconds: float) -> str:
    """Format seconds for the .RPP (6 decimals ≈ 0.04 samples at 44.1 kHz)."""
    return f"{float(seconds):.6f}"


def _wav_info(path: str) -> dict[str, Any]:
    """Validate a WAV file and return its facts.

    Raises ``FileNotFoundError`` when the file is missing and ``ValueError``
    when it is not a readable 44 100 Hz WAV. The studio never resamples
    mid-pipeline, so a wrong sample rate is a hard error, not a warning.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"WAV file not found: {path}")
    try:
        info = sf.info(str(file_path))
    except Exception as exc:
        raise ValueError(f"could not read WAV header of {path}: {exc}") from exc
    if info.samplerate != SAMPLE_RATE:
        raise ValueError(
            f"WAV {path} is {info.samplerate} Hz, not {SAMPLE_RATE} Hz — "
            "the studio never resamples mid-pipeline"
        )
    if info.frames <= 0:
        raise ValueError(f"WAV {path} is empty (0 frames)")
    return {
        "path": str(file_path),
        "file": file_path.name,
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "num_samples": int(info.frames),
        "duration_seconds": round(int(info.frames) / int(info.samplerate), 6),
    }


def _read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    """Read a WAV as mono float64 (stereo is mixed down by mean)."""
    _wav_info(path)  # validates existence + 44 100 Hz
    audio, sr = sf.read(str(path), dtype="float64", always_2d=False)
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    return np.ascontiguousarray(audio), int(sr)


def _role_for_stem(filename: str) -> str:
    """Map a stem filename to a mix role.

    Mirrors ``agents.engineer_agent._classify_stem`` (same needles, same
    fall-through to ``melodic``) plus the studio's own stem names:
    ``harmonic`` is melodic content, ``riser`` is an FX layer.
    """
    lowered = filename.lower()
    if "kick" in lowered:
        return "kick"
    if any(n in lowered for n in ("vocal", "vox", "voice", "choir", "harmony", "backing")):
        return "backing" if "backing" in lowered else "vocal"
    if any(n in lowered for n in ("bass", "sub", "808")):
        return "bass"
    if any(n in lowered for n in ("drum", "perc", "snare", "hihat", "hat", "cymbal", "clap", "tom", "beat")):
        return "drums"
    if "riser" in lowered:
        return "fx"
    return "melodic"


def _load_blueprint(blueprint_json: str | dict[str, Any]) -> ArrangementBlueprint:
    """Parse + validate a blueprint (JSON string or dict) against the schema.

    Raises ``ValueError`` with the Pydantic validation message on any
    schema violation (bad tiling, missing Drop, wrong sample rate, ...).
    """
    if isinstance(blueprint_json, str):
        try:
            data = json.loads(blueprint_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"blueprint_json is not valid JSON: {exc}") from exc
    elif isinstance(blueprint_json, dict):
        data = blueprint_json
    else:
        raise ValueError(
            f"blueprint_json must be a JSON string or dict, got {type(blueprint_json).__name__}"
        )
    try:
        return ArrangementBlueprint.model_validate(data)
    except Exception as exc:
        raise ValueError(f"blueprint failed validation: {exc}") from exc


def _collect_stems(stems_dir: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Collect the stem WAVs in ``stems_dir`` (validated, 44 100 Hz).

    Render outputs (``final_master_full.wav``, ``viral_hook_*.wav``) are
    skipped — they are not stems and get no track. Returns
    ``(stems, skipped_files)``; raises when the directory is missing or
    holds no stem WAVs at all.
    """
    dir_path = Path(stems_dir)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"stems directory not found: {stems_dir}")
    stems: list[dict[str, Any]] = []
    skipped: list[str] = []
    for wav_path in sorted(dir_path.glob("*.wav")):
        name = wav_path.name
        if name in _NON_STEM_NAMES or any(
            name.startswith(prefix) for prefix in _NON_STEM_PREFIXES
        ):
            skipped.append(name)
            continue
        info = _wav_info(str(wav_path))
        info["role"] = _role_for_stem(name)
        stems.append(info)
    if not stems:
        raise ValueError(
            f"no stem WAVs found in {stems_dir} "
            f"(skipped render outputs: {skipped or 'none'})"
        )
    return stems, skipped


def _validate_midi_file(path: str) -> dict[str, Any]:
    """Validate a .mid file parses with pretty_midi; return its facts."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"MIDI file not found: {path}")
    if file_path.suffix.lower() not in (".mid", ".midi"):
        raise ValueError(f"not a MIDI file (expected .mid/.midi): {path}")
    try:
        import pretty_midi
    except ImportError as exc:
        raise ImportError(
            "pretty_midi is not installed — MIDI validation cannot run. "
            "Install it with: pip install pretty_midi==0.2.11.post0"
        ) from exc
    try:
        pm = pretty_midi.PrettyMIDI(str(file_path))
    except Exception as exc:
        raise ValueError(f"could not parse MIDI file {path}: {exc}") from exc
    notes = [n for inst in pm.instruments for n in inst.notes]
    return {
        "path": str(file_path),
        "file": file_path.name,
        "duration_seconds": round(float(pm.get_end_time()), 6),
        "note_count": len(notes),
        "instruments": len(pm.instruments),
    }


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------
def _transcribe_basic_pitch(
    input_path: str, output_path: str, bpm: float
) -> dict[str, Any]:
    """Transcribe with Basic Pitch (only called when it is importable).

    The model is loaded inside a ``model_session("Basic Pitch")`` window
    (0.5 GiB registry estimate) and unloaded afterwards. Weights are read
    from ``models/basic-pitch/`` when that directory holds a model, else
    the library's bundled default model is used.
    """
    from basic_pitch.inference import predict  # type: ignore

    model_arg: Any = None
    if BASIC_PITCH_MODEL_DIR.is_dir() and any(BASIC_PITCH_MODEL_DIR.iterdir()):
        model_arg = str(BASIC_PITCH_MODEL_DIR)
    if _model_session is not None:
        ctx = _model_session("Basic Pitch")
    else:  # pragma: no cover - config import is guarded, never expected here
        from contextlib import nullcontext

        ctx = nullcontext()
    with ctx:
        if model_arg is not None:
            _model_output, midi_data, _note_events = predict(input_path, model_arg)
        else:
            _model_output, midi_data, _note_events = predict(input_path)
    midi_data.write(output_path)
    return _midi_facts(output_path, method="basic_pitch", model="basic_pitch")


def _transcribe_onset_fallback(
    input_path: str, output_path: str, bpm: float
) -> dict[str, Any]:
    """Onset-to-MIDI fallback (librosa onset detection + piptrack pitch).

    One note per detected onset segment: the pitch is the median piptrack
    pitch over the segment's high-energy frames, the start is quantized
    to the nearest 16th note at ``bpm``, the duration is the segment
    length (min 0.08 s), and the velocity follows the segment RMS. This
    is a transcription *estimate*, not a neural model — it is labelled
    ``onset_fallback`` in every return value and never called Basic Pitch.
    """
    import librosa
    import pretty_midi

    audio, sr = _read_wav_mono(input_path)
    if audio.size == 0:
        raise ValueError(f"audio file is empty: {input_path}")
    duration_s = audio.size / sr

    # Onset times (backtracked to the true attack) + the segment end.
    onset_times = librosa.onset.onset_detect(
        y=audio, sr=sr, units="time", backtrack=True, hop_length=512
    )
    onset_times = [float(t) for t in onset_times]
    if not onset_times or onset_times[0] > 0.05:
        onset_times = [0.0] + onset_times
    boundaries = onset_times + [duration_s]

    # Per-frame pitch + magnitude; frames are 512 samples apart.
    pitches, magnitudes = librosa.piptrack(
        y=audio, sr=sr, fmin=50.0, fmax=2000.0, hop_length=512
    )
    frame_times = librosa.frames_to_time(
        np.arange(pitches.shape[1]), sr=sr, hop_length=512
    )
    mag_floor = float(np.median(magnitudes)) if magnitudes.size else 0.0

    sixteenth_s = 60.0 / bpm / 4.0
    notes: list[pretty_midi.Note] = []
    skipped_segments = 0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < 0.03:
            continue
        frame_idx = np.where((frame_times >= start) & (frame_times < end))[0]
        if frame_idx.size == 0:
            skipped_segments += 1
            continue
        # Pitch per frame = the loudest piptrack bin; keep only frames
        # whose magnitude clears the track median (silence has no pitch).
        frame_pitches: list[float] = []
        frame_mags: list[float] = []
        for fi in frame_idx:
            mag_col = magnitudes[:, fi]
            best = int(np.argmax(mag_col))
            if mag_col[best] >= mag_floor:
                frame_pitches.append(float(pitches[best, fi]))
                frame_mags.append(float(mag_col[best]))
        if not frame_pitches:
            skipped_segments += 1
            continue
        freq = float(np.median(frame_pitches))
        if freq <= 0:
            skipped_segments += 1
            continue
        midi_note = int(round(float(librosa.hz_to_midi(freq))))
        if not 21 <= midi_note <= 108:
            skipped_segments += 1
            continue
        # Velocity from the segment RMS (40..110).
        seg = audio[int(start * sr) : max(int(end * sr), int(start * sr) + 1)]
        rms = float(np.sqrt(np.mean(seg**2))) if seg.size else 0.0
        velocity = int(min(110, max(40, 40 + rms * 300.0)))
        note_start = round(start / sixteenth_s) * sixteenth_s
        note_end = max(note_start + 0.08, end)
        note_end = min(note_end, duration_s)
        if note_end <= note_start:
            continue
        notes.append(
            pretty_midi.Note(
                velocity=velocity,
                pitch=midi_note,
                start=round(note_start, 6),
                end=round(note_end, 6),
            )
        )

    pm = pretty_midi.PrettyMIDI(resolution=480, initial_tempo=float(bpm))
    instrument = pretty_midi.Instrument(program=0, name="Transcription (onset fallback)")
    instrument.notes = notes
    pm.instruments.append(instrument)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_path))
    facts = _midi_facts(str(out_path), method="onset_fallback", model="onset_fallback (librosa piptrack + onset_detect)")
    facts["skipped_segments"] = skipped_segments
    return facts


def _midi_facts(output_path: str, method: str, model: str) -> dict[str, Any]:
    """Re-read a written MIDI file and return its verified facts."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(output_path)
    notes = [n for inst in pm.instruments for n in inst.notes]
    pitches = [n.pitch for n in notes]
    return {
        "output_path": str(Path(output_path)),
        "method": method,
        "model": model,
        "note_count": len(notes),
        "pitch_min": int(min(pitches)) if pitches else None,
        "pitch_max": int(max(pitches)) if pitches else None,
        "duration_seconds": round(float(pm.get_end_time()), 6),
        "valid_midi": True,
    }


# ---------------------------------------------------------------------------
# REAPER project writing
# ---------------------------------------------------------------------------
def _write_rpp(
    output_path: Path,
    bpm: float,
    tracks: list[dict[str, Any]],
) -> str:
    """Write a REAPER project file (plain-text .RPP) and verify its structure.

    ``tracks`` entries: ``{"name": str, "kind": "audio" | "midi",
    "file": str (relative path from the .RPP), "length_seconds": float,
    "position_seconds": float}``. Every FILE path must already exist
    (checked by the caller); this writer additionally asserts the
    written text parses back with balanced ``<...>`` blocks.
    """
    lines: list[str] = []
    lines.append('<REAPER_PROJECT 0.1 "7.0" 1730000000')
    lines.append('  RECORD_PATH "" ""')
    lines.append(f"  TEMPO {_fmt_seconds(bpm)} 4 4")
    lines.append("  PLAYRATE 1 0 0.25 4")
    for idx, track in enumerate(tracks, start=1):
        guid = _new_guid()
        lines.append(f"  <TRACK {guid}")
        lines.append(f"    NAME {_rpp_name(track['name'])}")
        lines.append(f"    TRACKID {guid}")
        lines.append("    NCHAN 2")
        lines.append("    VOLPAN 1 0 -1 -1 1")
        lines.append("    MUTESOLO 0 0 0")
        lines.append("    IPHASE 0")
        lines.append("    <ITEM")
        lines.append(f"      POSITION {_fmt_seconds(track['position_seconds'])}")
        lines.append("      SNAPOFFS 0")
        lines.append(f"      LENGTH {_fmt_seconds(track['length_seconds'])}")
        lines.append("      LOOP 1")
        lines.append("      ALLTAKES 0")
        lines.append("      FADEIN 1 0 0")
        lines.append("      FADEOUT 1 0 0")
        lines.append("      MUTE 0")
        lines.append("      SEL 0")
        lines.append(f"      IGUID {_new_guid()}")
        lines.append(f"      IID {idx}")
        lines.append("      VOLPAN 1 0 -1 -1 1")
        lines.append("      SOFFS 0")
        lines.append("      PLAYRATE 1 1 0 -1 0 0.0025")
        lines.append("      CHANMODE 0")
        lines.append(f"      GUID {_new_guid()}")
        if track["kind"] == "midi":
            lines.append("      <SOURCE MIDI")
        else:
            lines.append("      <SOURCE WAVE")
        lines.append(f'        FILE "{track["file"]}"')
        lines.append("      >")
        lines.append("    >")
        lines.append("  >")
    lines.append(">")
    text = "\n".join(lines) + "\n"

    # Structural self-check: every opened <BLOCK is closed by a lone '>'.
    opens = sum(1 for ln in lines if ln.lstrip().startswith("<"))
    closes = sum(1 for ln in lines if ln.strip() == ">")
    if opens != closes:
        raise ValueError(
            f"internal .RPP error: {opens} opened blocks vs {closes} closers"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return str(output_path)


def _parse_rpp(rpp_path: str) -> dict[str, Any]:
    """Parse a .RPP back into track/item/tempo facts (used by tests + tools).

    A small structural parser: it tracks ``<TRACK`` / ``<ITEM`` nesting
    and reads ``NAME``, ``TEMPO``, ``POSITION``, ``LENGTH`` and ``FILE``.
    Raises ``ValueError`` on unbalanced blocks or a missing TEMPO line.
    """
    lines = Path(rpp_path).read_text(encoding="utf-8").splitlines()
    opens = sum(1 for ln in lines if ln.lstrip().startswith("<"))
    closes = sum(1 for ln in lines if ln.strip() == ">")
    if opens != closes:
        raise ValueError(
            f".RPP {rpp_path} has unbalanced blocks: {opens} opens vs {closes} closes"
        )
    tempo: float | None = None
    tracks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    in_item = False
    for raw in lines:
        line = raw.strip()
        if line.startswith("TEMPO "):
            parts = line.split()
            tempo = float(parts[1])
        elif line.startswith("<TRACK"):
            current = {"name": None, "items": []}
            tracks.append(current)
        elif line.startswith("<ITEM"):
            in_item = True
            if current is not None:
                current["items"].append({})
        elif line == ">" and in_item:
            in_item = False
        elif line.startswith("NAME ") and current is not None and not in_item:
            name = line[len("NAME "):].strip()
            current["name"] = name[1:-1] if name.startswith('"') else name
        elif line.startswith("POSITION ") and in_item and current is not None:
            current["items"][-1]["position_seconds"] = float(line.split()[1])
        elif line.startswith("LENGTH ") and in_item and current is not None:
            current["items"][-1]["length_seconds"] = float(line.split()[1])
        elif line.startswith("FILE ") and in_item and current is not None:
            fname = line[len("FILE "):].strip()
            current["items"][-1]["file"] = fname[1:-1] if fname.startswith('"') else fname
    if tempo is None:
        raise ValueError(f".RPP {rpp_path} has no TEMPO line")
    return {"tempo_bpm": tempo, "track_count": len(tracks), "tracks": tracks}


def _build_stems_and_midi_layout(
    stems: list[dict[str, Any]],
    midi_files: list[dict[str, Any]],
    output_dir: Path,
    bpm: float,
) -> dict[str, Any]:
    """Copy stems + MIDI files into ``output_dir`` and write manifest.json.

    Files keep their original names; the manifest maps each file to its
    role, kind, duration and (for MIDI) note count. Every copied file is
    verified to exist afterwards — the layout never references a missing
    file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for stem in stems:
        src = Path(stem["path"])
        dest = output_dir / src.name
        if src.resolve() != dest.resolve():
            shutil.copy2(str(src), str(dest))
        if not dest.is_file():
            raise ValueError(f"failed to copy stem into layout: {src.name}")
        entries.append(
            {
                "role": stem["role"],
                "file": src.name,
                "kind": "audio",
                "sample_rate": stem["sample_rate"],
                "channels": stem["channels"],
                "num_samples": stem["num_samples"],
                "duration_seconds": stem["duration_seconds"],
            }
        )
    for midi in midi_files:
        src = Path(midi["path"])
        dest = output_dir / src.name
        if src.resolve() != dest.resolve():
            shutil.copy2(str(src), str(dest))
        if not dest.is_file():
            raise ValueError(f"failed to copy MIDI into layout: {src.name}")
        entries.append(
            {
                "role": "midi",
                "file": src.name,
                "kind": "midi",
                "duration_seconds": midi["duration_seconds"],
                "note_count": midi["note_count"],
            }
        )
    manifest = {
        "layout": "stems_and_midi",
        "sample_rate": SAMPLE_RATE,
        "bpm": float(bpm),
        "files": entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "files": entries,
    }


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------
@mcp.tool()
def audio_to_midi(
    input_path: str,
    output_path: str,
    bpm: float = 120.0,
    method: str = "auto",
) -> dict[str, Any]:
    """Transcribe an audio file to a valid MIDI file.

    Basic Pitch is the preferred transcriber (``method="basic_pitch"``),
    but it cannot run in this environment (see the module docstring) —
    with ``method="auto"`` (default) the tool falls back to a
    clearly-labelled onset-to-MIDI conversion (librosa onset detection +
    piptrack pitch, one note per onset segment, starts quantized to the
    nearest 16th note at ``bpm``) and reports ``method="onset_fallback"``
    in the return value. The fallback is never presented as Basic Pitch.

    Args:
        input_path: Input WAV file (44 100 Hz, mono or stereo).
        output_path: Destination ``.mid`` file.
        bpm: Tempo used for 16th-note quantization (must be > 0).
        method: ``"auto"`` (Basic Pitch if available, else fallback),
            ``"basic_pitch"`` (fail loudly when unavailable), or
            ``"onset_fallback"`` (force the fallback).

    Returns:
        Output path, transcription method/model, note count, pitch range,
        duration, and any warnings. The written MIDI is re-read and its
        facts verified before returning.
    """
    if bpm <= 0:
        raise ValueError(f"bpm must be > 0, got {bpm}")
    if method not in ("auto", "basic_pitch", "onset_fallback"):
        raise ValueError(
            f"method must be 'auto', 'basic_pitch' or 'onset_fallback', got {method!r}"
        )
    _wav_info(input_path)  # validates existence + 44 100 Hz
    if Path(output_path).suffix.lower() not in (".mid", ".midi"):
        raise ValueError(f"output_path must end in .mid/.midi, got {output_path!r}")
    status = _basic_pitch_status()
    use_basic_pitch = method == "basic_pitch" or (
        method == "auto" and status["available"]
    )
    if method == "basic_pitch" and not status["available"]:
        raise ImportError(
            "Basic Pitch is not available in this environment. " + status["reason"]
        )
    warnings: list[str] = []
    if use_basic_pitch:
        result = _transcribe_basic_pitch(input_path, output_path, bpm)
    else:
        if method == "auto":
            warnings.append(
                "Basic Pitch unavailable — used the onset-to-MIDI fallback. "
                + status["reason"]
            )
        result = _transcribe_onset_fallback(input_path, output_path, bpm)
    if result["note_count"] == 0:
        warnings.append("transcription produced 0 notes (silence or no clear onsets)")
    result["bpm"] = float(bpm)
    result["input_path"] = str(Path(input_path))
    result["warnings"] = warnings
    return result


@mcp.tool()
def export_daw_project(
    stems_dir: str,
    blueprint_json: str | dict[str, Any],
    output_path: str,
    bpm: float | None = None,
    midi_files: list[str] | None = None,
) -> dict[str, Any]:
    """Export a REAPER project file (.RPP) for a finished remix.

    One track per stem WAV in ``stems_dir`` (vocals, drums, bass, melodic,
    backing, kick_pattern, riser — whatever is present), each clip at
    POSITION 0 with LENGTH equal to the stem's exact duration in seconds
    (sample count / 44 100), tempo set to the blueprint BPM, plus one MIDI
    track per supplied ``.mid`` file. The stems + MIDI files are copied
    into ``stems_and_midi/`` next to the ``.RPP`` and referenced by
    relative paths, so the project opens in REAPER as a self-contained
    folder. ``blueprint_json`` is validated against the
    ``ArrangementBlueprint`` schema; a ``bpm`` argument, when given, must
    match the blueprint BPM.

    Args:
        stems_dir: Directory holding the stem WAV files (44 100 Hz).
        blueprint_json: Blueprint as a JSON string or dict.
        output_path: Destination ``.RPP`` file.
        bpm: Tempo override; when omitted the blueprint BPM is used.
        midi_files: Optional list of ``.mid`` file paths (one MIDI track
            each; each file must parse as valid MIDI).

    Returns:
        Output path, layout dir, manifest path, BPM, track list (name,
        role, file, position, length), MIDI tracks, skipped render
        outputs, and warnings.
    """
    blueprint = _load_blueprint(blueprint_json)
    tempo = float(bpm) if bpm is not None else float(blueprint.bpm)
    if tempo <= 0:
        raise ValueError(f"bpm must be > 0, got {tempo}")
    if bpm is not None and abs(float(bpm) - float(blueprint.bpm)) > 0.001:
        raise ValueError(
            f"bpm argument ({bpm}) disagrees with blueprint bpm "
            f"({blueprint.bpm}) — pass the blueprint tempo or omit bpm"
        )
    stems, skipped = _collect_stems(stems_dir)
    midi_infos: list[dict[str, Any]] = []
    for midi_path in midi_files or []:
        midi_infos.append(_validate_midi_file(midi_path))

    rpp_path = Path(output_path)
    if rpp_path.suffix.lower() != ".rpp":
        raise ValueError(f"output_path must end in .rpp, got {output_path!r}")
    layout_dir = rpp_path.parent / "stems_and_midi"
    layout = _build_stems_and_midi_layout(stems, midi_infos, layout_dir, tempo)

    # Tracks in canonical role order; stems are full-length files placed
    # at the track start (POSITION 0) with their exact duration.
    ordered = sorted(
        stems,
        key=lambda s: (
            _TRACK_ORDER.index(s["role"]) if s["role"] in _TRACK_ORDER else len(_TRACK_ORDER),
            s["file"],
        ),
    )
    tracks: list[dict[str, Any]] = []
    for stem in ordered:
        tracks.append(
            {
                "name": _ROLE_DISPLAY.get(stem["role"], stem["role"].title()),
                "kind": "audio",
                "file": f"stems_and_midi/{stem['file']}",
                "position_seconds": 0.0,
                "length_seconds": stem["duration_seconds"],
            }
        )
    for midi in midi_infos:
        midi_len = midi["duration_seconds"]
        if midi_len <= 0:
            midi_len = blueprint.duration_seconds
        tracks.append(
            {
                "name": f"MIDI — {Path(midi['file']).stem}",
                "kind": "midi",
                "file": f"stems_and_midi/{midi['file']}",
                "position_seconds": 0.0,
                "length_seconds": midi_len,
            }
        )

    written = _write_rpp(rpp_path, tempo, tracks)
    # Parse back: the .RPP must describe exactly what was written.
    parsed = _parse_rpp(written)
    if parsed["track_count"] != len(tracks):
        raise ValueError(
            f"internal .RPP error: wrote {len(tracks)} tracks, "
            f"parsed back {parsed['track_count']}"
        )
    return {
        "output_path": written,
        "layout_dir": layout["output_dir"],
        "manifest_path": layout["manifest_path"],
        "bpm": tempo,
        "blueprint_title": blueprint.title,
        "track_count": len(tracks),
        "tracks": [
            {
                "name": t["name"],
                "kind": t["kind"],
                "file": t["file"],
                "position_seconds": t["position_seconds"],
                "length_seconds": t["length_seconds"],
            }
            for t in tracks
        ],
        "midi_tracks": len(midi_infos),
        "skipped_render_outputs": skipped,
        "warnings": [],
    }


@mcp.tool()
def export_stems_and_midi(
    stems_dir: str,
    output_dir: str,
    midi_files: list[str] | None = None,
    bpm: float | None = None,
) -> dict[str, Any]:
    """Copy stems + MIDI files into the canonical stems_and_midi/ layout.

    Every stem WAV in ``stems_dir`` (44 100 Hz, validated) and every
    supplied ``.mid`` file (validated as parseable MIDI) is copied into
    ``output_dir`` under its original name, and a ``manifest.json`` is
    written listing each file with its role, kind, sample rate, duration
    and (for MIDI) note count. This is the layout the project spec
    requires for DAW import (``stems_and_midi/``).

    Args:
        stems_dir: Directory holding the stem WAV files (44 100 Hz).
        output_dir: Destination layout directory (created if needed).
        midi_files: Optional list of ``.mid`` file paths to include.
        bpm: Tempo to record in the manifest; when omitted, 120.0 is
            recorded and flagged in ``warnings`` as unconfirmed.

    Returns:
        Output dir, manifest path, BPM, file list, and warnings.
    """
    stems, skipped = _collect_stems(stems_dir)
    midi_infos: list[dict[str, Any]] = []
    for midi_path in midi_files or []:
        midi_infos.append(_validate_midi_file(midi_path))
    warnings: list[str] = []
    tempo = 120.0 if bpm is None else float(bpm)
    if bpm is None:
        warnings.append(
            "no bpm given — manifest records 120.0 as unconfirmed; "
            "pass bpm to record the real tempo"
        )
    elif tempo <= 0:
        raise ValueError(f"bpm must be > 0, got {tempo}")
    layout = _build_stems_and_midi_layout(
        stems, midi_infos, Path(output_dir), tempo
    )
    return {
        "output_dir": layout["output_dir"],
        "manifest_path": layout["manifest_path"],
        "bpm": tempo,
        "files": layout["files"],
        "skipped_render_outputs": skipped,
        "warnings": warnings,
    }


def main() -> None:
    """Run the DAW export MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
