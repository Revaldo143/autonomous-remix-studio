"""Direct-call smoke tests for mcp_servers/dsp_mastering_server.py.

Generates 1-second test WAVs with NumPy/soundfile, then calls the MCP tool
functions directly (no MCP transport involved) and prints the results.
Run from the project root with the project venv:

    .venv/bin/python tests/test_dsp_mastering_server.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp_servers import dsp_mastering_server as server  # noqa: E402

SR = 44100
TMP = PROJECT_ROOT / "tests" / "_tmp"
TMP.mkdir(parents=True, exist_ok=True)


def call_tool(tool, *args, **kwargs):
    """Call a @mcp.tool() function directly, tolerating the wrapper."""
    try:
        return tool(*args, **kwargs)
    except TypeError:
        return tool.fn(*args, **kwargs)


def make_test_wavs() -> dict[str, Path]:
    """1-second stereo mix + mono vocal-like stem, 44.1 kHz, 24-bit."""
    t = np.arange(SR, dtype=np.float64) / SR
    # Stereo mix: 440 Hz left, 554 Hz right, plus kick-like pulses.
    left = 0.4 * np.sin(2 * np.pi * 440.0 * t)
    right = 0.4 * np.sin(2 * np.pi * 554.0 * t)
    for kick_ms in (0, 250, 500, 750):
        start = int(kick_ms * SR / 1000)
        decay = np.exp(-np.arange(0, 2000) / 300.0)
        left[start:start + 2000] += 0.5 * decay
        right[start:start + 2000] += 0.5 * decay
    mix = np.stack([left, right], axis=1)
    mix = np.clip(mix, -0.95, 0.95)
    mix_path = TMP / "test_mix.wav"
    sf.write(str(mix_path), mix, SR, subtype="PCM_24")

    # Mono vocal-like stem: 880 Hz with a slow amplitude envelope.
    vocal = 0.5 * np.sin(2 * np.pi * 880.0 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t))
    vocal_path = TMP / "test_vocal.wav"
    sf.write(str(vocal_path), vocal, SR, subtype="PCM_24")

    # Distinct reference track for the Matchering path: 2 s stereo,
    # different tonal content (220/330 Hz + slow tremolo) — must NOT be
    # the same file as the mix (Matchering rejects target == reference).
    t2 = np.arange(2 * SR, dtype=np.float64) / SR
    env2 = 0.5 + 0.3 * np.sin(2 * np.pi * 0.5 * t2)
    ref = np.stack([
        0.35 * np.sin(2 * np.pi * 220.0 * t2) * env2,
        0.35 * np.sin(2 * np.pi * 330.0 * t2) * env2,
    ], axis=1)
    ref_path = TMP / "test_reference.wav"
    sf.write(str(ref_path), ref, SR, subtype="PCM_24")
    return {"mix": mix_path, "vocal": vocal_path, "reference": ref_path}


def check_pcm24(path: Path) -> str:
    info = sf.info(str(path))
    assert info.subtype == "PCM_24", f"{path}: expected PCM_24, got {info.subtype}"
    return f"{info.samplerate} Hz, {info.channels} ch, {info.subtype}, {info.duration:.3f}s"


def main() -> None:
    paths = make_test_wavs()
    mix = str(paths["mix"])
    vocal = str(paths["vocal"])

    print("=== 1. analyze_audio ===")
    analysis = call_tool(server.analyze_audio, mix)
    print(json.dumps(analysis, indent=2))
    assert analysis["duration_seconds"] == 1.0
    assert analysis["channels"] == 2
    assert np.isfinite(analysis["lufs"])

    print("\n=== 2. apply_sidechain (kicks at 0/250/500/750 ms, depth 6 dB) ===")
    sc_out = str(TMP / "test_mix_sidechained.wav")
    sc = call_tool(
        server.apply_sidechain, mix, [0.0, 250.0, 500.0, 750.0], sc_out,
        depth_db=6.0, release_ms=120.0,
    )
    print(json.dumps(sc, indent=2))
    print("output:", check_pcm24(Path(sc_out)))
    assert sc["peak_dbfs_after"] < sc["peak_dbfs_before"]

    print("\n=== 3. master_track (streaming, -14 LUFS) ===")
    m_out = str(TMP / "test_mix_master_streaming.wav")
    mastered = call_tool(server.master_track, mix, None, -14.0, m_out)
    print(json.dumps(mastered, indent=2))
    print("output:", check_pcm24(Path(m_out)))
    assert abs(mastered["metrics_after"]["lufs"] - (-14.0)) < 0.6, mastered["metrics_after"]
    assert mastered["metrics_after"]["dbtp"] <= -1.0 + 1e-6, mastered["metrics_after"]

    print("\n=== 4. master_track (club, -9 LUFS) ===")
    club_out = str(TMP / "test_mix_master_club.wav")
    club = call_tool(server.master_track, mix, None, -9.0, club_out)
    print(json.dumps(club, indent=2))
    assert abs(club["metrics_after"]["lufs"] - (-9.0)) < 0.6, club["metrics_after"]
    assert club["metrics_after"]["dbtp"] <= -1.0 + 1e-6

    print("\n=== 5. master_track with reference (Matchering path) ===")
    ref_out = str(TMP / "test_mix_master_ref.wav")
    ref = call_tool(server.master_track, mix, str(paths["reference"]), -14.0, ref_out)
    print(json.dumps(ref, indent=2))
    # Matchering is installed in the project venv (requirements pin 2.0.6),
    # so a distinct reference must actually be matched end to end.
    assert ref["reference_matched"] is True, ref
    assert ref["warnings"] == [], ref["warnings"]
    assert abs(ref["metrics_after"]["lufs"] - (-14.0)) < 0.6
    assert ref["metrics_after"]["dbtp"] <= -1.0 + 1e-6
    print("output:", check_pcm24(Path(ref_out)))

    print("\n=== 5b. master_track with reference == target (graceful fallback) ===")
    same_out = str(TMP / "test_mix_master_ref_same.wav")
    same = call_tool(server.master_track, mix, mix, -14.0, same_out)
    print(json.dumps(same, indent=2))
    # Matchering rejects target == reference; the server must fall back to
    # the loudness path, never fail mastering.
    assert same["reference_matched"] is False
    assert any("matchering" in w.lower() for w in same["warnings"]), same["warnings"]
    assert abs(same["metrics_after"]["lufs"] - (-14.0)) < 0.6

    print("\n=== 6. align_beat_grid (120 BPM, downbeats 0/500 ms) ===")
    ab_out = str(TMP / "test_mix_aligned.wav")
    aligned = call_tool(server.align_beat_grid, mix, 120.0, [0.0, 500.0], ab_out)
    print(json.dumps(aligned, indent=2))
    print("output:", check_pcm24(Path(ab_out)))

    print("\n=== 7. carve_vocals (bands=8, reduction 3 dB) ===")
    cv_out = str(TMP / "test_mix_carved.wav")
    carved = call_tool(server.carve_vocals, mix, vocal, cv_out, bands=8, reduction_db=3.0)
    print(json.dumps(carved, indent=2))
    print("output:", check_pcm24(Path(cv_out)))
    # D2 regression: carve_vocals must report the same LUFS as analyze_audio
    # for the same file (both now measure in core layout). The old code read
    # the stereo file as 44,100 channels of 2 samples and under-reported.
    assert abs(carved["lufs_before"] - analysis["lufs"]) < 0.1, (
        f"lufs_before {carved['lufs_before']} != analyze_audio "
        f"{analysis['lufs']} for the same file"
    )

    print("\n=== 8. time_stretch (rate 2.0 -> 0.5 s) ===")
    ts_out = str(TMP / "test_mix_stretched.wav")
    stretched = call_tool(server.time_stretch, mix, 2.0, ts_out)
    print(json.dumps(stretched, indent=2))
    assert stretched["duration_seconds"] == 0.5, stretched
    print("output:", check_pcm24(Path(ts_out)))

    print("\n=== 9. spatial_pan (azimuth -45 deg, left should dominate) ===")
    sp_out = str(TMP / "test_mix_panned.wav")
    panned = call_tool(server.spatial_pan, mix, -45.0, sp_out)
    print(json.dumps(panned, indent=2))
    data, _ = sf.read(sp_out)
    rms_l = float(np.sqrt(np.mean(data[:, 0] ** 2)))
    rms_r = float(np.sqrt(np.mean(data[:, 1] ** 2)))
    print(f"RMS L={rms_l:.4f} R={rms_r:.4f}")
    assert rms_l > rms_r
    print("output:", check_pcm24(Path(sp_out)))

    print("\n=== 10. apply_pedalboard_chain ===")
    pb_out = str(TMP / "test_mix_pedalboard.wav")
    pb = call_tool(
        server.apply_pedalboard_chain, mix, pb_out,
        compressor_threshold_db=-18.0, reverb_mix=0.15, highpass_hz=30.0,
    )
    print(json.dumps(pb, indent=2))
    print("output:", check_pcm24(Path(pb_out)))
    # D2 regression (second site): pedalboard loudness must agree with
    # analyze_audio for the same file, within 0.1 LUFS.
    assert abs(pb["lufs_before"] - analysis["lufs"]) < 0.1, (
        f"lufs_before {pb['lufs_before']} != analyze_audio "
        f"{analysis['lufs']} for the same file"
    )

    print("\n=== 11. MCP tool surface (async list_tools) ===")
    tools = asyncio.run(server.mcp.list_tools())
    names = sorted(t.name for t in tools)
    print("registered tools:", names)
    expected = sorted([
        "align_beat_grid", "apply_sidechain", "carve_vocals", "time_stretch",
        "spatial_pan", "analyze_audio", "master_track", "apply_pedalboard_chain",
    ])
    assert names == expected, f"tool mismatch: {names} != {expected}"

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
