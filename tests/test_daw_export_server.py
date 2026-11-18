"""Direct-call smoke tests for mcp_servers/daw_export_server.py.

Generates test WAVs with NumPy/soundfile, then calls the MCP tool
functions directly (no MCP transport involved) and prints the results.
Run from the project root with the project venv:

    .venv/bin/python tests/test_daw_export_server.py

Basic Pitch is NOT installed in this environment (it cannot be: its
dependency chain rebuilds the pinned numpy against a Python-3.12-
incompatible source tree), so the audio_to_midi tests assert the
honestly-labelled onset-to-MIDI fallback, never Basic Pitch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp_servers import daw_export_server as server  # noqa: E402

SR = 44100
TMP = PROJECT_ROOT / "tests" / "_tmp" / "daw"
TMP.mkdir(parents=True, exist_ok=True)

CHECKS = {"passed": 0, "failed": 0}


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record one check and print it."""
    if condition:
        CHECKS["passed"] += 1
        print(f"  PASS  {name}" + (f" — {detail}" if detail else ""))
    else:
        CHECKS["failed"] += 1
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))


def call_tool(tool, *args, **kwargs):
    """Call a @mcp.tool() function directly, tolerating the wrapper."""
    try:
        return tool(*args, **kwargs)
    except TypeError:
        return tool.fn(*args, **kwargs)


def make_melody_wav(path: Path, bpm: float = 120.0) -> list[tuple[float, int]]:
    """10 s synth melody: 8 notes, 1.25 s each, sine + crisp gaps.

    Notes (MIDI): C5 E5 G5 A5 G5 E5 D5 C5 — onsets at 0, 1.25, ..., 8.75 s.
    Returns the expected (onset_seconds, midi_note) list.
    """
    notes_midi = [72, 76, 79, 81, 79, 76, 74, 72]
    note_dur = 1.25
    total = len(notes_midi) * note_dur
    audio = np.zeros(int(total * SR), dtype=np.float64)
    expected: list[tuple[float, int]] = []
    for i, midi_note in enumerate(notes_midi):
        freq = 440.0 * 2.0 ** ((midi_note - 69) / 12.0)
        start = int(i * note_dur * SR)
        end = int((i + 1) * note_dur * SR)
        n = end - start
        t = np.arange(n, dtype=np.float64) / SR
        # 10 ms attack, fade out over the last 50 ms, 30 ms silence gap.
        env = np.ones(n)
        attack = int(0.010 * SR)
        env[:attack] = np.linspace(0.0, 1.0, attack)
        release = int(0.050 * SR)
        env[-release:] = np.linspace(1.0, 0.0, release)
        gap = int(0.030 * SR)
        env[-gap:] = 0.0
        audio[start:end] = 0.5 * np.sin(2 * np.pi * freq * t) * env
        expected.append((i * note_dur, midi_note))
    sf.write(str(path), audio, SR, subtype="PCM_24")
    return expected


def make_stem_wav(path: Path, duration_s: float, freq: float) -> None:
    """One full-length stem: a sine at ``freq`` Hz with a slow tremolo."""
    n = int(duration_s * SR)
    t = np.arange(n, dtype=np.float64) / SR
    audio = 0.4 * np.sin(2 * np.pi * freq * t) * (0.7 + 0.3 * np.sin(2 * np.pi * 0.5 * t))
    stereo = np.stack([audio, audio], axis=1)
    sf.write(str(path), stereo, SR, subtype="PCM_24")


def fake_blueprint() -> dict:
    """32 s blueprint at 120 BPM: 4 sections x 4 bars, hooks 15 s + 30 s."""
    return {
        "title": "Test Remix",
        "track_duration_ms": 32000,
        "bpm": 120.0,
        "key": "D major",
        "sections": [
            {"name": "intro", "start_ms": 0, "end_ms": 8000, "energy": 0.2,
             "bpm": 120.0, "key": "D major", "bars": 4, "genre_fusion": ["kuthu", "phonk"]},
            {"name": "build_up", "start_ms": 8000, "end_ms": 16000, "energy": 0.6,
             "bpm": 120.0, "key": "D major", "bars": 4, "genre_fusion": ["kuthu", "phonk"]},
            {"name": "drop", "start_ms": 16000, "end_ms": 24000, "energy": 0.9,
             "bpm": 120.0, "key": "D major", "bars": 4, "genre_fusion": ["kuthu", "phonk"]},
            {"name": "outro", "start_ms": 24000, "end_ms": 32000, "energy": 0.4,
             "bpm": 120.0, "key": "D major", "bars": 4, "genre_fusion": ["kuthu", "phonk"]},
        ],
        "viral_hooks": [
            {"duration_s": 15, "start_ms": 2000, "end_ms": 17000, "retention_score": 0.9},
            {"duration_s": 30, "start_ms": 2000, "end_ms": 32000, "retention_score": 0.88},
        ],
        "sample_rate": 44100,
    }


def test_audio_to_midi_melody() -> Path:
    print("== audio_to_midi on a 10 s synth melody ==")
    melody_path = TMP / "melody.wav"
    expected = make_melody_wav(melody_path)
    midi_path = TMP / "melody.mid"
    result = call_tool(server.audio_to_midi, str(melody_path), str(midi_path), bpm=120.0)
    check("method is the honest onset fallback (not Basic Pitch)",
          result["method"] == "onset_fallback", result["method"])
    check("fallback warning present",
          any("fallback" in w.lower() for w in result.get("warnings", [])))
    check("MIDI file written and valid", midi_path.is_file() and result["valid_midi"])
    check("note count covers the melody (>= 6 of 8)",
          result["note_count"] >= 6, f"got {result['note_count']}")
    check("pitches in the melody range (60..84)",
          result["pitch_min"] is not None and 60 <= result["pitch_min"]
          and result["pitch_max"] <= 84,
          f"{result['pitch_min']}..{result['pitch_max']}")
    # Timing: every expected onset must have a transcribed note within 0.15 s.
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    starts = sorted(n.start for inst in pm.instruments for n in inst.notes)
    matched = sum(
        1 for onset, _note in expected
        if any(abs(s - onset) <= 0.15 for s in starts)
    )
    check("note starts match expected onsets within 0.15 s",
          matched >= 6, f"{matched}/8 onsets matched; starts={starts}")
    check("MIDI duration ≈ 10 s", abs(result["duration_seconds"] - 10.0) <= 0.3,
          f"{result['duration_seconds']} s")
    return midi_path


def test_audio_to_midi_silence() -> None:
    print("== audio_to_midi on 2 s of silence ==")
    silence_path = TMP / "silence.wav"
    sf.write(str(silence_path), np.zeros(int(2 * SR)), SR, subtype="PCM_24")
    result = call_tool(
        server.audio_to_midi, str(silence_path), str(TMP / "silence.mid"), bpm=120.0
    )
    check("silence transcribes to 0 notes without crashing",
          result["note_count"] == 0 and result["valid_midi"])
    check("0-note warning present",
          any("0 notes" in w for w in result.get("warnings", [])))


def test_export_daw_project(midi_path: Path) -> None:
    print("== export_daw_project on a fake blueprint + 3 stems ==")
    stems_dir = TMP / "stems3"
    stems_dir.mkdir(parents=True, exist_ok=True)
    make_stem_wav(stems_dir / "vocals.wav", 32.0, 440.0)
    make_stem_wav(stems_dir / "drums.wav", 32.0, 110.0)
    make_stem_wav(stems_dir / "bass.wav", 32.0, 55.0)
    # A render output must NOT get a track.
    make_stem_wav(stems_dir / "final_master_full.wav", 32.0, 220.0)

    rpp_path = TMP / "test_project" / "project.RPP"
    result = call_tool(
        server.export_daw_project,
        str(stems_dir), fake_blueprint(), str(rpp_path),
    )
    check("RPP written", rpp_path.is_file())
    check("BPM from blueprint", result["bpm"] == 120.0, str(result["bpm"]))
    check("track count is 3 (master output skipped)",
          result["track_count"] == 3, str(result["track_count"]))
    check("master listed as skipped",
          "final_master_full.wav" in result["skipped_render_outputs"])
    check("stems_and_midi layout written",
          Path(result["layout_dir"]).is_dir()
          and Path(result["manifest_path"]).is_file())
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    check("manifest lists 3 audio files",
          len(manifest["files"]) == 3, str(len(manifest["files"])))
    check("manifest sample rate 44100", manifest["sample_rate"] == 44100)
    check("manifest durations are 32.0 s",
          all(abs(f["duration_seconds"] - 32.0) < 0.001 for f in manifest["files"]))

    # Parse the .RPP back: track count, tempo, clip positions, file existence.
    parsed = server._parse_rpp(str(rpp_path))
    check("parsed tempo is 120.0", parsed["tempo_bpm"] == 120.0,
          str(parsed["tempo_bpm"]))
    check("parsed track count is 3", parsed["track_count"] == 3)
    names = [t["name"] for t in parsed["tracks"]]
    check("track names are Vocals/Drums/Bass in role order",
          names == ["Vocals", "Drums", "Bass"], str(names))
    positions_ok = all(
        abs(item["position_seconds"] - 0.0) < 1e-6
        for t in parsed["tracks"] for item in t["items"]
    )
    lengths_ok = all(
        abs(item["length_seconds"] - 32.0) < 1e-6
        for t in parsed["tracks"] for item in t["items"]
    )
    check("all clips at POSITION 0", positions_ok)
    check("all clips LENGTH 32.0 s", lengths_ok)
    files_ok = all(
        (rpp_path.parent / item["file"]).is_file()
        for t in parsed["tracks"] for item in t["items"]
    )
    check("every FILE in the .RPP exists on disk (relative paths)", files_ok)

    # With a MIDI track: 4 tracks, manifest has the .mid.
    print("== export_daw_project with a MIDI track ==")
    rpp2_path = TMP / "test_project_midi" / "project.RPP"
    result2 = call_tool(
        server.export_daw_project,
        str(stems_dir), fake_blueprint(), str(rpp2_path),
        midi_files=[str(midi_path)],
    )
    check("track count is 4 with MIDI", result2["track_count"] == 4,
          str(result2["track_count"]))
    check("one MIDI track reported", result2["midi_tracks"] == 1)
    manifest2 = json.loads(Path(result2["manifest_path"]).read_text())
    check("manifest has 3 audio + 1 midi",
          len(manifest2["files"]) == 4
          and sum(1 for f in manifest2["files"] if f["kind"] == "midi") == 1)
    parsed2 = server._parse_rpp(str(rpp2_path))
    midi_items = [t for t in parsed2["tracks"] if t["name"].startswith("MIDI")]
    check("parsed MIDI track present", len(midi_items) == 1,
          str([t["name"] for t in parsed2["tracks"]]))
    check("MIDI FILE exists on disk",
          (rpp2_path.parent / midi_items[0]["items"][0]["file"]).is_file()
          if midi_items else False)


def test_export_stems_and_midi() -> None:
    print("== export_stems_and_midi ==")
    stems_dir = TMP / "stems3"
    out_dir = TMP / "layout_only"
    result = call_tool(
        server.export_stems_and_midi, str(stems_dir), str(out_dir), bpm=126.05
    )
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    check("manifest bpm recorded", manifest["bpm"] == 126.05, str(manifest["bpm"]))
    check("manifest durations correct",
          all(abs(f["duration_seconds"] - 32.0) < 0.001 for f in manifest["files"]))
    check("manifest sample rate 44100", manifest["sample_rate"] == 44100)
    check("all layout files exist on disk",
          all((out_dir / f["file"]).is_file() for f in manifest["files"]))


def test_invalid_inputs() -> None:
    print("== invalid inputs fail cleanly ==")
    # Missing audio file.
    try:
        call_tool(server.audio_to_midi, str(TMP / "nope.wav"), str(TMP / "x.mid"))
        check("missing audio raises", False, "no error raised")
    except FileNotFoundError as exc:
        check("missing audio raises FileNotFoundError", True, str(exc)[:60])

    # 48 kHz WAV rejected (studio never resamples).
    wrong_sr = TMP / "wrong_sr.wav"
    sf.write(str(wrong_sr), np.zeros(48000), 48000, subtype="PCM_24")
    try:
        call_tool(server.audio_to_midi, str(wrong_sr), str(TMP / "x.mid"))
        check("48 kHz WAV raises", False, "no error raised")
    except ValueError as exc:
        check("48 kHz WAV raises ValueError", "44100" in str(exc), str(exc)[:80])

    # Bad blueprint JSON.
    stems_dir = TMP / "stems3"
    try:
        call_tool(
            server.export_daw_project, str(stems_dir), "{not json",
            str(TMP / "bad.RPP"),
        )
        check("bad blueprint JSON raises", False, "no error raised")
    except ValueError as exc:
        check("bad blueprint JSON raises ValueError", True, str(exc)[:60])

    # Blueprint that fails schema validation (no Drop section).
    bad_bp = fake_blueprint()
    bad_bp["sections"] = [s for s in bad_bp["sections"] if s["name"] != "drop"]
    try:
        call_tool(
            server.export_daw_project, str(stems_dir), bad_bp,
            str(TMP / "bad2.RPP"),
        )
        check("blueprint without Drop raises", False, "no error raised")
    except ValueError as exc:
        check("blueprint without Drop raises ValueError", True, str(exc)[:60])

    # BPM argument disagreeing with the blueprint.
    try:
        call_tool(
            server.export_daw_project, str(stems_dir), fake_blueprint(),
            str(TMP / "bad3.RPP"), bpm=100.0,
        )
        check("bpm mismatch raises", False, "no error raised")
    except ValueError as exc:
        check("bpm mismatch raises ValueError", True, str(exc)[:60])

    # Missing stems dir.
    try:
        call_tool(
            server.export_stems_and_midi, str(TMP / "no_such_dir"),
            str(TMP / "layout_x"),
        )
        check("missing stems dir raises", False, "no error raised")
    except FileNotFoundError as exc:
        check("missing stems dir raises FileNotFoundError", True, str(exc)[:60])

    # audio_to_midi with method="basic_pitch" fails loudly, not silently.
    melody_path = TMP / "melody.wav"
    try:
        call_tool(
            server.audio_to_midi, str(melody_path), str(TMP / "bp.mid"),
            method="basic_pitch",
        )
        check("basic_pitch method raises when unavailable", False, "no error raised")
    except ImportError as exc:
        check("basic_pitch method raises ImportError (honest)",
              "Basic Pitch is not available" in str(exc))

    # A non-MIDI file passed as midi_files.
    try:
        call_tool(
            server.export_daw_project, str(stems_dir), fake_blueprint(),
            str(TMP / "bad4.RPP"), midi_files=[str(melody_path)],
        )
        check("non-MIDI midi_file raises", False, "no error raised")
    except ValueError as exc:
        check("non-MIDI midi_file raises ValueError", True, str(exc)[:60])


def main() -> int:
    print("DAW export server tests — project root:", PROJECT_ROOT)
    midi_path = test_audio_to_midi_melody()
    test_audio_to_midi_silence()
    test_export_daw_project(midi_path)
    test_export_stems_and_midi()
    test_invalid_inputs()
    print(
        f"\nRESULT: {CHECKS['passed']} passed, {CHECKS['failed']} failed"
    )
    return 0 if CHECKS["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
