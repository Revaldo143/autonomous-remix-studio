"""Direct-call tests for mcp_servers/stem_theory_server.py.

Runs the MCP tool functions directly (no MCP transport involved) and
prints the results. Run from the project root with the project venv:

    .venv/bin/python tests/test_stem_theory_server.py

NOTE: the separation tests load the 527 MB BS-RoFormer checkpoint and
run CPU inference — the 30 s synthetic separation takes a few minutes
on this box, and the real-song smoke test (first 30 s of Veezha Veera)
takes a few more. That is the honest cost of a real neural model on
CPU; the harmony / validation tests run in seconds.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp_servers import stem_theory_server as server  # noqa: E402

SR = 44100
TMP = PROJECT_ROOT / "tests" / "_tmp"
TMP.mkdir(parents=True, exist_ok=True)

REAL_SONG = PROJECT_ROOT / "outputs" / "yt-test" / "veezha-veera-44100.wav"


def call_tool(tool, *args, **kwargs):
    """Call a @mcp.tool() function directly, tolerating the wrapper."""
    try:
        return tool(*args, **kwargs)
    except TypeError:
        return tool.fn(*args, **kwargs)


def check_pcm24_exact(path: Path, n_samples: int) -> str:
    """Assert 24-bit / 44 100 Hz / stereo / exact sample count."""
    info = sf.info(str(path))
    assert info.subtype == "PCM_24", f"{path}: expected PCM_24, got {info.subtype}"
    assert info.samplerate == SR, f"{path}: sr {info.samplerate}"
    assert info.channels == 2, f"{path}: channels {info.channels}"
    assert info.frames == n_samples, (
        f"{path}: {info.frames} samples != input {n_samples}"
    )
    return f"{info.samplerate} Hz, {info.channels} ch, {info.subtype}, {info.frames} samples"


def make_synthetic_mix(duration_s: float = 30.0) -> Path:
    """30 s stereo mix at 120 BPM with separable layers at different
    levels, so the drums stem must contain the kick train:

    * kick: 55 Hz sine bursts on every beat (0.5 s grid), loudest layer.
    * bass: 110 Hz sine, continuous.
    * pad: 440 + 554 + 659 Hz (A major triad), continuous.
    * vocal-like: 880 Hz with 5 Hz vibrato + slow tremolo, panned left.
    """
    n = int(duration_s * SR)
    t = np.arange(n, dtype=np.float64) / SR
    mix = np.zeros((n, 2), dtype=np.float64)

    # Kick train: 120 BPM, 150 ms bursts, 55 Hz with pitch drop.
    burst_len = int(0.15 * SR)
    bt = np.arange(burst_len, dtype=np.float64) / SR
    freq = 50.0 + 100.0 * np.exp(-bt / 0.03)
    phase = 2.0 * np.pi * np.cumsum(freq) / SR
    burst = np.sin(phase) * np.exp(-bt / 0.045)
    for start in range(0, n, SR // 2):
        end = min(n, start + burst_len)
        mix[start:end, 0] += 0.9 * burst[: end - start]
        mix[start:end, 1] += 0.9 * burst[: end - start]

    # Bass: 110 Hz sine.
    bass = 0.35 * np.sin(2.0 * np.pi * 110.0 * t)
    mix[:, 0] += bass
    mix[:, 1] += bass

    # Plucked arpeggio (A major: 440/554/659/880 Hz), 250 ms apart, with
    # harmonics and fast decay — realistic pitched content for the
    # model's "other" stem (steady sine pads get lumped into "bass" by
    # this checkpoint, so the mix uses onsets instead).
    notes = [440.0, 554.37, 659.25, 880.0]
    pluck_len = int(0.22 * SR)
    pt = np.arange(pluck_len, dtype=np.float64) / SR
    for i, start in enumerate(range(0, n, SR // 4)):
        f0 = notes[i % len(notes)]
        end = min(n, start + pluck_len)
        seg_t = pt[: end - start]
        pluck = (
            np.sin(2.0 * np.pi * f0 * seg_t)
            + 0.5 * np.sin(2.0 * np.pi * 2.0 * f0 * seg_t)
            + 0.25 * np.sin(2.0 * np.pi * 3.0 * f0 * seg_t)
        ) * np.exp(-seg_t / 0.07)
        mix[start:end, 0] += 0.18 * pluck
        mix[start:end, 1] += 0.18 * pluck

    # Hats: short noise bursts on the off-beats.
    rng = np.random.default_rng(7)
    hat_len = int(0.05 * SR)
    for start in range(SR // 4, n, SR // 2):
        end = min(n, start + hat_len)
        noise = rng.standard_normal(end - start) * np.exp(
            -np.arange(end - start) / (hat_len / 4.0)
        )
        mix[start:end, 0] += 0.15 * noise
        mix[start:end, 1] += 0.15 * noise

    # Vocal-like: 880 Hz vibrato, tremolo, left-leaning.
    vib = 880.0 * (1.0 + 0.006 * np.sin(2.0 * np.pi * 5.0 * t))
    tremolo = 0.6 + 0.4 * np.sin(2.0 * np.pi * 3.0 * t)
    vocal = 0.30 * np.sin(2.0 * np.pi * np.cumsum(vib) / SR) * tremolo
    mix[:, 0] += vocal
    mix[:, 1] += 0.5 * vocal

    mix = np.clip(mix, -0.95, 0.95)
    path = TMP / "stem_test_mix_30s.wav"
    sf.write(str(path), mix, SR, subtype="PCM_24")
    return path


def make_c_major_progression(duration_s: float = 8.0) -> Path:
    """8 s of C–F–G–C (2 s each), sine-stack triads, stereo."""
    chords = [(261.63, 329.63, 392.00), (174.61, 220.00, 261.63),
              (196.00, 246.94, 293.66), (261.63, 329.63, 392.00)]
    n = int(duration_s * SR)
    t = np.arange(n, dtype=np.float64) / SR
    mix = np.zeros((n, 2), dtype=np.float64)
    for i, triad in enumerate(chords):
        start = int(i * 2.0 * SR)
        end = int(min(n, (i + 1) * 2.0 * SR))
        seg = np.zeros(end - start)
        for f in triad:
            seg += 0.15 * np.sin(2.0 * np.pi * f * t[start:end])
        mix[start:end, 0] += seg
        mix[start:end, 1] += seg
    mix = np.clip(mix, -0.95, 0.95)
    path = TMP / "stem_test_c_major.wav"
    sf.write(str(path), mix, SR, subtype="PCM_24")
    return path


def kick_energy_ratio(stem_path: Path) -> tuple[float, float]:
    """Low-band energy in kick windows vs off-beat windows.

    Kicks sit on the 0.5 s grid; measure RMS of the <= 200 Hz content
    in 100 ms windows centred on beats vs centred halfway between
    beats. A real drums stem has a much higher on-beat ratio than the
    mix or a wrong stem.
    """
    from scipy.signal import butter, sosfiltfilt

    audio, sr = sf.read(str(stem_path), always_2d=True, dtype="float64")
    mono = audio.mean(axis=1)
    sos = butter(4, 200.0, btype="lowpass", fs=sr, output="sos")
    low = sosfiltfilt(sos, mono)
    on, off = [], []
    for beat in range(0, 30):
        c_on = int((beat * 0.5 + 0.05) * sr)
        c_off = int((beat * 0.5 + 0.30) * sr)
        w = int(0.08 * sr)
        on.append(float(np.sqrt(np.mean(low[c_on : c_on + w] ** 2))))
        off.append(float(np.sqrt(np.mean(low[c_off : c_off + w] ** 2))))
    on_rms = float(np.mean(on))
    off_rms = float(np.mean(off))
    return on_rms, (on_rms / off_rms if off_rms > 0 else float("inf"))


def main() -> None:
    print("=== 1. separate_stems on 30 s synthetic mix ===")
    mix_path = make_synthetic_mix()
    n_samples = int(30.0 * SR)
    t0 = time.time()
    result = call_tool(
        server.separate_stems,
        str(mix_path),
        model="BS-RoFormer",
        output_dir=str(TMP / "stem_test_mix_30s_stems"),
    )
    elapsed = time.time() - t0
    print(f"separation took {elapsed:.1f} s")
    print(json.dumps(result, indent=2))
    assert result["model"] == "BS-RoFormer"
    stems = result["stems"]
    assert set(stems) == {"vocals", "backing", "drums", "bass", "melodic"}, stems
    assert result["derivation"]["vocals"] == "bs-roformer"
    assert result["derivation"]["drums"] == "bs-roformer"
    assert result["derivation"]["bass"] == "bs-roformer"
    assert "hpss" in result["derivation"]["melodic"]
    assert "hpss" in result["derivation"]["backing"]
    for name, path in stems.items():
        print(f"OK {name}: {check_pcm24_exact(Path(path), n_samples)}")
    # The drums stem must actually contain the kick train — measured,
    # not just "file exists".
    on_rms, ratio = kick_energy_ratio(Path(stems["drums"]))
    print(f"drums stem: on-beat low-band RMS {on_rms:.4f}, on/off ratio {ratio:.2f}x")
    assert ratio > 2.0, f"drums stem does not contain the kick train (ratio {ratio:.2f})"
    # melodic + backing must reconstruct the model's "other" stem.
    other, _ = sf.read(result["other_stem"], always_2d=True, dtype="float64")
    melodic, _ = sf.read(stems["melodic"], always_2d=True, dtype="float64")
    backing, _ = sf.read(stems["backing"], always_2d=True, dtype="float64")
    recon_err = float(np.max(np.abs((melodic + backing) - other)))
    print(f"melodic + backing reconstructs 'other' with max abs error {recon_err:.2e}")
    assert recon_err < 1e-4, f"HPSS split does not reconstruct 'other': {recon_err}"

    print("\n=== 2. analyze_harmony on known C-major progression ===")
    cmaj = make_c_major_progression()
    harmony = call_tool(server.analyze_harmony, str(cmaj))
    print(json.dumps(harmony, indent=2))
    assert harmony["key"] == "C major", harmony["key"]
    assert harmony["key_confidence"] >= 0.7, harmony["key_confidence"]
    found = [c["chord"] for c in harmony["chords"]]
    print("chords found:", found)
    assert "C" in found and "F" in found and "G" in found, found

    print("\n=== 3. detect_raga_tala on C-major progression ===")
    raga = call_tool(server.detect_raga_tala, str(cmaj))
    print(json.dumps(raga, indent=2))
    # C major triads = Shankarabharanam scale (all white keys) at tonic C.
    assert raga["raga"] in ("Shankarabharanam", "unknown"), raga["raga"]
    assert raga["tala"] in ("unknown", "Adi (8-beat)", "Rupakam (6-beat)",
                            "Misra Chapu (7-beat)", "Khanda Chapu (5-beat)")

    print("\n=== 4. invalid inputs fail cleanly ===")
    try:
        call_tool(server.separate_stems, str(TMP / "does_not_exist.wav"))
    except FileNotFoundError as exc:
        print("missing file -> FileNotFoundError:", exc)
    else:
        raise AssertionError("missing file did not raise")
    wrong_sr = TMP / "wrong_sr.wav"
    sf.write(str(wrong_sr), np.zeros((48000, 2)), 48000, subtype="PCM_24")
    try:
        call_tool(server.separate_stems, str(wrong_sr))
    except ValueError as exc:
        print("wrong sample rate -> ValueError:", exc)
    else:
        raise AssertionError("wrong sample rate did not raise")
    try:
        call_tool(server.analyze_harmony, str(TMP / "does_not_exist.wav"))
    except FileNotFoundError as exc:
        print("harmony missing file -> FileNotFoundError:", exc)
    else:
        raise AssertionError("harmony missing file did not raise")

    print("\n=== 5. real-song smoke test: first 30 s of Veezha Veera ===")
    assert REAL_SONG.is_file(), f"real song missing: {REAL_SONG}"
    clip, sr = sf.read(str(REAL_SONG), always_2d=True, dtype="float64",
                       start=0, stop=int(30.0 * SR))
    clip_path = TMP / "veezha_veera_30s.wav"
    sf.write(str(clip_path), clip, SR, subtype="PCM_24")
    t0 = time.time()
    real = call_tool(
        server.separate_stems,
        str(clip_path),
        model="BS-RoFormer",
        output_dir=str(TMP / "veezha_veera_30s_stems"),
    )
    elapsed = time.time() - t0
    print(f"real-song separation took {elapsed:.1f} s")
    print(json.dumps(real, indent=2))
    for name, path in real["stems"].items():
        info = sf.info(path)
        print(f"OK {name}: {info.frames} samples, {info.subtype}, "
              f"RMS {real['rms_dbfs'][name]} dBFS")
        assert info.frames == int(30.0 * SR)
        assert info.subtype == "PCM_24"
    real_harmony = call_tool(server.analyze_harmony, str(clip_path))
    print("real-song harmony:", json.dumps(real_harmony, indent=2))
    real_raga = call_tool(server.detect_raga_tala, str(clip_path))
    print("real-song raga/tala:", json.dumps(real_raga, indent=2))

    print("\nALL STEM-THEORY TESTS PASSED")


if __name__ == "__main__":
    main()
