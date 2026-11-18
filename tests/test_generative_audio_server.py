"""Direct-call tests for mcp_servers/generative_audio_server.py.

Calls the MCP tool functions directly (no MCP transport involved) and
measures the audio: exact sample lengths, 24-bit 44.1 kHz stereo
format, onset energy on the beat grid, rising riser energy, and
in-scale chroma for melodies. Run from the project root with the
project venv:

    .venv/bin/python tests/test_generative_audio_server.py

Generated WAVs go to /tmp/gen-test/ and are deleted at the end of the
run (/tmp is a 512 MB tmpfs — keep it tidy).

Engine honesty: every tool reports which engine actually produced the
audio ("musicgen-melody" or "dsp-fallback"). Measurements that only the
deterministic DSP path can guarantee (exact beat-grid onsets, strict
in-scale chroma) are asserted strictly for dsp-fallback and relaxed
to "real audio exists" checks for the model path — a model cannot be
expected to land onsets sample-exactly on the grid.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp_servers import generative_audio_server as server  # noqa: E402

SR = 44100
TMP = Path("/tmp/gen-test")
TMP.mkdir(parents=True, exist_ok=True)


def call_tool(tool, *args, **kwargs):
    """Call a @mcp.tool() function directly, tolerating the wrapper."""
    try:
        return tool(*args, **kwargs)
    except TypeError:
        return tool.fn(*args, **kwargs)


def check_pcm24(path: Path) -> str:
    info = sf.info(str(path))
    assert info.samplerate == SR, f"{path}: expected {SR} Hz, got {info.samplerate}"
    assert info.channels == 2, f"{path}: expected stereo, got {info.channels}ch"
    assert info.subtype == "PCM_24", f"{path}: expected PCM_24, got {info.subtype}"
    return f"{info.samplerate} Hz, {info.channels} ch, {info.subtype}, {info.duration:.3f}s"


def read_mono(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=True, dtype="float64")
    return audio.mean(axis=1), sr


def beat_grid_ratio(mono: np.ndarray, sr: int, bpm: float, n_beats: int) -> float:
    """Median low-band RMS at beat centers / median at off-beat centers.

    Kicks are the only <= 200 Hz content in the DSP beat, so a real
    tempo-locked beat shows a large ratio; a flat or random file ~1.
    """
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, 200.0, btype="lowpass", fs=sr, output="sos")
    low = sosfiltfilt(sos, mono)
    beat_len = int(round(sr * 60.0 / bpm))
    win = max(1, int(round(0.04 * sr)))  # 40 ms windows
    beat_rms: list[float] = []
    off_rms: list[float] = []
    for beat in range(n_beats):
        center = beat * beat_len
        lo, hi = max(0, center - win // 2), min(len(low), center + win // 2)
        beat_rms.append(float(np.sqrt(np.mean(low[lo:hi] ** 2))) if hi > lo else 0.0)
        off_center = center + beat_len // 2
        lo, hi = max(0, off_center - win // 2), min(len(low), off_center + win // 2)
        off_rms.append(float(np.sqrt(np.mean(low[lo:hi] ** 2))) if hi > lo else 0.0)
    med_beat = float(np.median(beat_rms))
    med_off = float(np.median(off_rms))
    return med_beat / med_off if med_off > 0 else float("inf")


def chunk_rms(mono: np.ndarray, n_chunks: int = 8) -> np.ndarray:
    """RMS per equal chunk of the signal."""
    chunks = np.array_split(mono, n_chunks)
    return np.array(
        [float(np.sqrt(np.mean(c**2))) if c.size else 0.0 for c in chunks]
    )


def chroma_in_scale_fraction(mono: np.ndarray, sr: int, pitch_classes: set[int]) -> float:
    """Fraction of mean chroma energy on in-scale pitch classes (0-11).

    Chroma leaks into neighbouring semitone bins, so even a perfectly
    in-scale sine melody reads ~0.68 here — it is reported as a sanity
    signal, not the strict check (see pitch_in_scale_fraction).
    """
    import librosa

    chroma = librosa.feature.chroma_cqt(y=mono, sr=sr)
    mean_chroma = chroma.mean(axis=1)
    total = float(np.sum(mean_chroma))
    if total <= 0:
        return 0.0
    in_scale = float(sum(mean_chroma[pc] for pc in pitch_classes))
    return in_scale / total


def pitch_in_scale_fraction(mono: np.ndarray, sr: int, pitch_classes: set[int], bpm: float, bars: int) -> float:
    """Fraction of 8th-note steps whose dominant pitch is in-scale.

    The strict in-scale check: for each 8th-note step, take the middle
    50% of the step, find the strongest spectral peak (the pluck
    fundamental dominates its harmonics), and check its pitch class.
    Steps quieter than 5% of the loudest step are rests and are skipped.
    This is robust where frame-wise YIN is not — overlapping pluck
    tails confuse a monophonic tracker, but the dominant spectral peak
    per step is unambiguous.
    """
    step_len = int(round(sr * 60.0 / bpm / 2.0))
    total_steps = bars * 8
    step_rms: list[float] = []
    windows: list[np.ndarray] = []
    for step in range(total_steps):
        start = step * step_len
        if start >= len(mono):
            break
        end = min(len(mono), start + step_len)
        seg = mono[start:end]
        if seg.size == 0:
            continue
        step_rms.append(float(np.sqrt(np.mean(seg**2))))
        windows.append(seg)
    if not step_rms:
        return 0.0
    peak_rms = max(step_rms)
    if peak_rms <= 0:
        return 0.0
    in_scale = 0
    counted = 0
    for seg, rms in zip(windows, step_rms):
        if rms < 0.05 * peak_rms:
            continue  # rest
        # Middle 50% of the step: the newest note dominates there.
        lo = seg.size // 4
        hi = max(lo + 1, (3 * seg.size) // 4)
        windowed = seg[lo:hi] * np.hanning(hi - lo)
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(hi - lo, d=1.0 / sr)
        spectrum[freqs < 60.0] = 0.0
        peak_idx = int(np.argmax(spectrum))
        peak_freq = float(freqs[peak_idx])
        if peak_freq <= 0 or spectrum[peak_idx] <= 0:
            continue
        midi = int(round(69 + 12 * np.log2(peak_freq / 440.0)))
        counted += 1
        if midi % 12 in pitch_classes:
            in_scale += 1
    if counted == 0:
        return 0.0
    return in_scale / counted


def expect_value_error(label: str, fn, *args, **kwargs) -> None:
    """Assert fn raises ValueError with a readable message."""
    try:
        fn(*args, **kwargs)
    except ValueError as exc:
        print(f"  OK {label}: ValueError({exc})")
        return
    raise AssertionError(f"{label}: expected ValueError, call succeeded")


def main() -> None:
    print("=== 0. engine_status ===")
    status = call_tool(server.engine_status)
    print(json.dumps(status, indent=2))
    engine = status["engine"]
    assert engine in ("musicgen-melody", "dsp-fallback"), status
    print(f"active engine for this run: {engine}")

    print("\n=== 1. generate_beat(120, 4, 'kuthu') — exact 352800 samples ===")
    beat_out = str(TMP / "beat_120_4_kuthu.wav")
    beat = call_tool(server.generate_beat, 120.0, 4, "kuthu", beat_out)
    print(json.dumps(beat, indent=2))
    print("output:", check_pcm24(Path(beat_out)))
    assert beat["num_samples"] == 352800, beat
    assert beat["duration_seconds"] == 8.0, beat
    assert beat["engine"] == engine, beat
    mono, _ = read_mono(beat_out)
    peak = float(np.max(np.abs(mono)))
    print(f"peak amplitude: {peak:.3f}")
    assert peak > 0.1, "beat is (near-)silent"
    if engine == "dsp-fallback":
        ratio = beat_grid_ratio(mono, SR, 120.0, 16)
        print(f"beat/off-beat low-band RMS ratio: {ratio:.2f}")
        assert ratio > 3.0, f"no real onset energy on the beat grid: ratio {ratio}"
    else:
        print("model engine: grid alignment not asserted (model output)")

    print("\n=== 1b. generate_beat producer convention (key, no output_path) ===")
    beat2 = call_tool(
        server.generate_beat, bpm=120.0, key="D major", bars=2, style="kuthu"
    )
    print(json.dumps(beat2, indent=2))
    assert beat2["num_samples"] == 176400, beat2
    assert Path(beat2["output_path"]).is_file(), beat2
    print("output:", check_pcm24(Path(beat2["output_path"])))
    assert "outputs/generative" in beat2["output_path"], beat2

    print("\n=== 1c. generate_beat(126.05, 4) — non-round BPM exact length ===")
    beat3_out = str(TMP / "beat_126_4.wav")
    beat3 = call_tool(server.generate_beat, 126.05, 4, "phonk", beat3_out)
    expected = int(round(16 * 60.0 / 126.05 * SR))
    print(f"num_samples={beat3['num_samples']} expected={expected}")
    assert beat3["num_samples"] == expected, beat3
    print("output:", check_pcm24(Path(beat3_out)))

    print("\n=== 2. generate_riser(4.0 s) — exact 176400 samples, rising ===")
    riser_out = str(TMP / "riser_4s.wav")
    riser = call_tool(server.generate_riser, duration_s=4.0, style="phonk", output_path=riser_out)
    print(json.dumps(riser, indent=2))
    print("output:", check_pcm24(Path(riser_out)))
    assert riser["num_samples"] == 176400, riser
    mono, _ = read_mono(riser_out)
    rms = chunk_rms(mono, 8)
    print("chunk RMS:", [round(float(v), 4) for v in rms])
    assert rms[-1] > 3.0 * rms[0], f"riser does not rise: first {rms[0]:.4f} last {rms[-1]:.4f}"
    if engine == "dsp-fallback":
        assert bool(np.all(np.diff(rms) >= -1e-9)), f"DSP riser not monotonic: {rms}"
    else:
        # A KEPT model riser must have passed the server's rise gate
        # (>= 3.0x last/first chunk RMS); a flat model output is
        # discarded for the exact DSP riser and never arrives here as
        # musicgen-melody. (One model run read 0.83x — flat, not a
        # riser — which is why the gate exists.)
        reported_ratio = riser.get("rise_ratio")
        assert isinstance(reported_ratio, (int, float)), riser
        assert reported_ratio >= 3.0, f"kept model riser below rise gate: {reported_ratio}"
        print(f"server-reported rise_ratio: {reported_ratio:.2f}x")

    print("\n=== 2b. generate_riser producer convention (duration_ms) ===")
    riser2 = call_tool(server.generate_riser, duration_ms=2000, style="phonk")
    print(json.dumps(riser2, indent=2))
    assert riser2["num_samples"] == 88200, riser2
    assert Path(riser2["output_path"]).is_file(), riser2

    print("\n=== 3. generate_melody(120, 2, 'D major') — exact, in-scale ===")
    mel_out = str(TMP / "melody_d_major.wav")
    mel = call_tool(server.generate_melody, 120.0, 2, "D major", mel_out)
    print(json.dumps(mel, indent=2))
    print("output:", check_pcm24(Path(mel_out)))
    assert mel["num_samples"] == 176400, mel
    assert mel["scale"] == "D major", mel
    mono, _ = read_mono(mel_out)
    assert float(np.max(np.abs(mono))) > 0.1, "melody is (near-)silent"
    # D major pitch classes: D=2, E=4, F#=6, G=7, A=9, B=11, C#=1.
    in_scale_frac = chroma_in_scale_fraction(mono, SR, {1, 2, 4, 6, 7, 9, 11})
    print(f"chroma in D major: {in_scale_frac:.3f} (leakage-limited, sanity only)")
    pitch_frac = pitch_in_scale_fraction(mono, SR, {1, 2, 4, 6, 7, 9, 11}, 120.0, 2)
    print(f"dominant pitch in D major (per 8th-note step): {pitch_frac:.3f}")
    # Chroma is the gate for BOTH engines (chance 7/12 = 0.583 for a
    # 7-note scale; leakage caps even a perfect line at ~0.76-0.86).
    assert in_scale_frac > 0.6, f"melody chroma far from D major: {in_scale_frac}"
    # The server also reports its own chroma on the delivered audio
    # — it must be present and in range.
    reported = mel.get("scale_adherence_chroma")
    assert isinstance(reported, (int, float)) and 0.0 <= reported <= 1.0, mel
    print(f"server-reported scale_adherence_chroma: {reported:.3f}")
    if engine == "dsp-fallback":
        # DSP is exact by construction: every note is scale-quantized,
        # so the per-step dominant pitch IS the melody and must be in
        # scale (measured 1.000 on the DSP run).
        assert pitch_frac > 0.85, f"DSP melody leaves D major: {pitch_frac}"
    else:
        # Model engine: the per-step dominant-pitch check does NOT apply
        # — on polyphonic model audio the dominant peak in every step is
        # a sustained bass note (measured: a G2 drone in all 16 steps),
        # so the metric measures the bass, not the melody. The chroma
        # gate above is the scale check for the model. And the server's
        # scale gate must have held: a KEPT model loop reads >= 0.65
        # chroma (a below-bar loop is discarded for the exact DSP
        # fallback, so it never arrives here as musicgen-melody).
        assert reported >= 0.65, f"kept model loop below scale gate: {reported}"
        print("model engine: dominant-pitch check skipped (bass-dominated)")

    print("\n=== 4. invalid inputs fail cleanly ===")
    expect_value_error("bpm 30", server.generate_beat, 30.0, 4, "kuthu", str(TMP / "x.wav"))
    expect_value_error("bpm 300", server.generate_beat, 300.0, 4, "kuthu", str(TMP / "x.wav"))
    expect_value_error("bars 0", server.generate_beat, 120.0, 0, "kuthu", str(TMP / "x.wav"))
    expect_value_error("bars -2", server.generate_beat, 120.0, -2, "kuthu", str(TMP / "x.wav"))
    expect_value_error("empty style", server.generate_beat, 120.0, 4, "  ", str(TMP / "x.wav"))
    expect_value_error("riser no duration", server.generate_riser)
    expect_value_error("riser 0 ms", server.generate_riser, duration_ms=0)
    expect_value_error("riser -1 s", server.generate_riser, duration_s=-1.0)
    expect_value_error("melody bad scale", server.generate_melody, 120.0, 2, "X major", str(TMP / "x.wav"))
    expect_value_error("melody bad mode", server.generate_melody, 120.0, 2, "D locrian", str(TMP / "x.wav"))

    print("\n=== 5. MCP tool surface (async list_tools) ===")
    tools = asyncio.run(server.mcp.list_tools())
    names = sorted(t.name for t in tools)
    print("registered tools:", names)
    expected = sorted([
        "generate_beat", "generate_riser", "generate_melody",
        "engine_status", "unload_model",
    ])
    assert names == expected, f"tool mismatch: {names} != {expected}"

    print("\n=== 6. unload_model ===")
    unloaded = call_tool(server.unload_model)
    print(json.dumps(unloaded, indent=2))
    assert unloaded["model_loaded_in_process"] is False

    print("\n=== 7. torchvision compat shim (broken 0.29.2 vs torch 2.15.0) ===")
    # The shim is load-bearing for the model path: transformers'
    # image_utils imports torchvision at module level, and the venv's
    # torchvision 0.29.2 cannot register its ops against torch 2.15.0.
    # After the shim, the MusicGen classes must import cleanly.
    server._ensure_torchvision_compat()
    from transformers import (  # noqa: E402
        MusicgenMelodyForConditionalGeneration,
        MusicgenMelodyProcessor,
    )

    print("MusicgenMelody classes import OK after shim:", MusicgenMelodyForConditionalGeneration.__name__, MusicgenMelodyProcessor.__name__)

    print("\nALL TESTS PASSED")
    print(f"engine used for this run: {engine}")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Keep /tmp tidy: remove generated test audio.
        shutil.rmtree(TMP, ignore_errors=True)
        print(f"(cleaned {TMP})")
