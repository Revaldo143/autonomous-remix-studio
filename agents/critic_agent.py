"""Multimodal RLHF Audio Critic agent for the Autonomous Remix Studio.

The critic is the self-correction loop of the LangGraph workflow. It loads
the Engineer agent's master WAV and grades it with deterministic DSP
metrics — no LLM is required for a verdict:

* integrated LUFS vs the mix target, true peak vs the -1.0 dBTP ceiling,
  and any clipping (``core.dsp_utils``);
* stereo phase correlation between the L/R channels;
* transition roughness at the blueprint's section boundaries;
* vocal-masking estimate in the 300 Hz-3.4 kHz band;
* beat-sync drift: median downbeat-to-beat offset vs the bpm grid,
  whose phase is the circular mean of all detected downbeats (D12:
  anchoring at the first detection alone false-rejected on-grid tracks
  whose first attack is slow — the flux peak lags the attack by 30-50
  ms — while the rest of the train is on-grid).

The verdict is a :class:`agents.state.CriticReport`. ``"approved"`` is
only legal with an empty ``issues`` list (enforced by the model itself).
On rejection with iterations left (``iteration_count < 3``) the critic
derives concrete corrective ``mix_params`` for the re-render and the
graph routes back to the Engineer; otherwise the run ends.

An optional ``llm`` callable (``Callable[[dict], str]``) may be supplied
for a future Qwen2-Audio / Gemini multimodal verdict. The LLM text is
appended to the report summary as a note only — the metric verdict above
is authoritative and complete without it.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.state import (  # noqa: E402
    MAX_CRITIC_ITERATIONS,
    SAMPLE_RATE,
    CriticReport,
    RemixAgentState,
    SectionType,
)
from core.dsp_utils import detect_clipping, measure_loudness, ms_to_samples  # noqa: E402

__all__ = ["critic_node", "route_after_critic", "make_critic_node"]

# ---------------------------------------------------------------------------
# Thresholds (all in the units the checks report in)
# ---------------------------------------------------------------------------

DEFAULT_TARGET_LUFS: float = -14.0
LUFS_TOLERANCE_DB: float = 1.0
TRUE_PEAK_CEILING_DBTP: float = -1.0
PHASE_CORRELATION_MIN: float = 0.2
# Transition window: one full bar (4 beats) either side of a boundary.
# A window must span at least one period of the dominant pulse train so
# a periodic pulse contributes the same energy on both sides regardless
# of transient phase (D9: 200 ms windows landed on different transient
# phases on impulsive material — a decaying hat tail vs a full kick read
# +12.3 dB where 500 ms / 1 s windows read 0.0 dB — and the loop could
# not correct it, because a crossfade cannot change which transient
# phase falls in each 200 ms window). At 120 BPM one bar is 2000 ms;
# the window is clamped to the adjacent sections' lengths, and falls
# back to 1000 ms when the bpm is unknown.
TRANSITION_WINDOW_BEATS: float = 4.0
TRANSITION_WINDOW_FALLBACK_MS: float = 1000.0
TRANSITION_JUMP_DB: float = 12.0
# Residual jump a corrective section-gain trim aims for (half the limit).
TRANSITION_TARGET_DB: float = 6.0
MASKING_BAND_HZ: tuple[float, float] = (300.0, 3400.0)
MASKING_LIMIT_DB: float = 6.0
BEAT_OFFSET_LIMIT_MS: float = 30.0
MASKING_WINDOW_MS: int = 100

_SPECTRAL_BANDS: tuple[tuple[str, float, float], ...] = (
    ("sub", 20.0, 60.0),
    ("low", 60.0, 250.0),
    ("low_mid", 250.0, 500.0),
    ("mid", 500.0, 2000.0),
    ("presence", 2000.0, 6000.0),
    ("brilliance", 6000.0, 20000.0),
)

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_master(path: str) -> tuple[np.ndarray, int]:
    """Load a master WAV as ``(channels, samples)`` float64 plus its rate.

    Raises ``FileNotFoundError`` / ``ValueError`` on a missing, empty or
    unreadable file — a critic report on audio that does not exist would
    be a fabrication, so the node fails loudly instead.
    """
    master = Path(path)
    if not master.is_file():
        raise FileNotFoundError(f"master_path does not exist: {path}")
    audio, sr = sf.read(str(master), always_2d=True, dtype="float64")
    if audio.shape[0] == 0:
        raise ValueError(f"master is empty (0 samples): {path}")
    # soundfile layout is (samples, channels); the DSP core is (channels, samples).
    return np.ascontiguousarray(audio.T), int(sr)


def _load_mono(path: str, sr: int) -> np.ndarray | None:
    """Load a stem as mono float64 at ``sr``; ``None`` when unreadable."""
    stem = Path(path)
    if not stem.is_file():
        return None
    audio, stem_sr = sf.read(str(stem), always_2d=True, dtype="float64")
    if audio.shape[0] == 0 or stem_sr != sr:
        return None
    return np.ascontiguousarray(audio.mean(axis=1))


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _phase_correlation(audio_2d: np.ndarray) -> tuple[float, str | None]:
    """Pearson correlation between L and R over the whole file.

    Returns ``(correlation, warning)``. Mono input reports 1.0 with a
    warning. Identical stereo channels (dual-mono) are a warning, not a
    blocking issue — they are mono-compatible, just not wide.
    """
    if audio_2d.shape[0] < 2:
        return 1.0, "master is mono — no stereo image to evaluate"
    left, right = audio_2d[0], audio_2d[1]
    if np.max(np.abs(left - right)) < 1e-9:
        return 1.0, "stereo channels are identical (dual-mono) — no stereo width"
    std_l, std_r = float(np.std(left)), float(np.std(right))
    if std_l < _EPS or std_r < _EPS:
        return 0.0, "one stereo channel is silent"
    corr = float(np.corrcoef(left, right)[0, 1])
    return corr, None


def _transition_roughness(
    audio_2d: np.ndarray, sr: int, blueprint: Any, bpm: float | None = None
) -> list[tuple[str, str, float]]:
    """Flag section boundaries where RMS jumps > 12 dB without a Drop.

    Returns one ``(issue_text, louder_section_name, jump_db)`` tuple per
    flagged boundary, so the caller can derive a corrective section
    gain for the louder section.

    RMS is measured over one full bar (4 beats) either side of each
    boundary, clamped to the adjacent sections' lengths (a section is
    at least one bar, so the window always fits). A bar-length window
    is phase-robust: a periodic pulse train contributes the same
    energy on both sides regardless of which transient phase each
    window edge lands on. The old 200 ms windows did not — on sparse
    impulsive material a decaying hat tail on one side vs a full kick
    on the other read +12.3 dB where 500 ms and 1 s windows both read
    0.0 dB, i.e. there was no real level step at all (D9). A real
    sustained step still shows up: every window on the loud side is
    loud and every window on the quiet side is quiet, so a genuine
    ~15 dB section-level step reads ~15 dB at any window length.

    A boundary counts as a Drop boundary when the section starting or
    ending there is a Drop — drops are allowed to jump.
    """
    flagged: list[tuple[str, str, float]] = []
    if blueprint is None:
        return flagged
    if bpm is not None and bpm > 0:
        window_ms = TRANSITION_WINDOW_BEATS * 60_000.0 / float(bpm)
    else:
        window_ms = TRANSITION_WINDOW_FALLBACK_MS
    sections = blueprint.sections
    for prev, cur in zip(sections, sections[1:]):
        boundary = ms_to_samples(prev.end_ms, sr)
        if boundary <= 0 or boundary >= audio_2d.shape[1]:
            continue
        # Clamp the window to the shorter adjacent section so the
        # measurement never bleeds across a second boundary.
        w_ms = min(window_ms, prev.end_ms - prev.start_ms, cur.end_ms - cur.start_ms)
        if w_ms <= 0:
            continue
        window = ms_to_samples(w_ms, sr)
        before = audio_2d[:, max(0, boundary - window) : boundary]
        after = audio_2d[:, boundary : boundary + window]
        rms_before = float(np.sqrt(np.mean(before**2))) if before.size else 0.0
        rms_after = float(np.sqrt(np.mean(after**2))) if after.size else 0.0
        if rms_before < _EPS or rms_after < _EPS:
            continue
        jump_db = 20.0 * math.log10(rms_after / rms_before)
        is_drop_boundary = (
            prev.name == SectionType.DROP or cur.name == SectionType.DROP
        )
        if abs(jump_db) > TRANSITION_JUMP_DB and not is_drop_boundary:
            louder = cur.name.value if jump_db > 0 else prev.name.value
            flagged.append(
                (
                    f"Transition at {prev.end_ms} ms "
                    f"({prev.name.value} -> {cur.name.value}) jumps "
                    f"{jump_db:+.1f} dB (limit +/-{TRANSITION_JUMP_DB:.0f} dB, "
                    "no Drop at this boundary)",
                    louder,
                    jump_db,
                )
            )
    return flagged


def _vocal_masking(
    audio_2d: np.ndarray, sr: int, stems: dict[str, str]
) -> list[str]:
    """Estimate vocal masking in the 300 Hz-3.4 kHz band.

    The vocal stem is located by a key containing ``"vocal"``. In each
    100 ms window where the vocal is active, the instrumental band energy
    is estimated as master band energy minus vocal band energy; the
    median ratio over active windows is the masking figure. An issue is
    raised when the instrumental exceeds the vocal by more than 6 dB.
    Returns ``[]`` (no issue) when there is no vocal stem, no active
    vocal window, or the stem cannot be read.
    """
    vocal_key = next((k for k in stems if "vocal" in k.lower()), None)
    if vocal_key is None:
        return []
    vocal = _load_mono(stems[vocal_key], sr)
    if vocal is None:
        return [
            f"Vocal stem {stems[vocal_key]!r} could not be read "
            "at the master sample rate — masking check skipped"
        ]
    master_mono = audio_2d.mean(axis=0)
    n = min(vocal.shape[0], master_mono.shape[0])
    if n == 0:
        return []
    vocal, master_mono = vocal[:n], master_mono[:n]

    win = ms_to_samples(MASKING_WINDOW_MS, sr)
    freqs = np.fft.rfftfreq(win, d=1.0 / sr)
    band = (freqs >= MASKING_BAND_HZ[0]) & (freqs <= MASKING_BAND_HZ[1])
    ratios_db: list[float] = []
    for start in range(0, n - win + 1, win):
        v_seg = vocal[start : start + win]
        if float(np.sqrt(np.mean(v_seg**2))) < 0.01:  # vocal silent here
            continue
        v_spec = np.abs(np.fft.rfft(v_seg * np.hanning(win))) ** 2
        m_spec = np.abs(np.fft.rfft(master_mono[start : start + win] * np.hanning(win))) ** 2
        e_vocal = float(np.sum(v_spec[band]))
        e_master = float(np.sum(m_spec[band]))
        if e_vocal < _EPS:
            continue
        e_instr = max(e_master - e_vocal, _EPS)
        ratios_db.append(10.0 * math.log10(e_instr / e_vocal))
    if not ratios_db:
        return []
    median_db = float(np.median(ratios_db))
    if median_db > MASKING_LIMIT_DB:
        return [
            f"Vocal masking: instrumental exceeds vocal by {median_db:.1f} dB "
            f"in the {MASKING_BAND_HZ[0]:.0f} Hz-"
            f"{MASKING_BAND_HZ[1]/1000:.1f} kHz band "
            f"(limit {MASKING_LIMIT_DB:.0f} dB)"
        ]
    return []


def _low_band_flux(mono: np.ndarray, sr: int, hop_length: int = 256) -> np.ndarray:
    """Positive spectral flux of the <= 200 Hz STFT bins, one value/frame.

    Computed on the STFT magnitudes of the *original* signal, restricted
    to low bins: a steady tone has constant magnitude in its bin, so it
    contributes ~zero flux however loud it is, while a kick transient
    jumps the low-bin magnitudes. (Low-passing in the time domain first
    would be wrong: two steady tones beating against each other modulate
    the low-passed envelope and look like onsets — that is exactly what
    made the first version of this check false-reject pads.)
    """
    import librosa

    spec = np.abs(librosa.stft(mono, n_fft=2048, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low = spec[freqs <= 200.0]
    diffs = np.diff(low, axis=1, prepend=low[:, :1])
    return np.sum(np.maximum(diffs, 0.0), axis=0)


_ONSET_HOP_LENGTH = 256


def _pick_peaks_numpy(
    env: np.ndarray, threshold: float, min_gap_frames: int
) -> np.ndarray:
    """Greedy strongest-first peak picker (scipy-free fallback).

    Local maxima of ``env`` at or above ``threshold``, de-duplicated so
    no two kept peaks are closer than ``min_gap_frames``: candidates are
    considered strongest-first, so within a window the tallest peak
    wins. Same contract as the ``scipy.signal.find_peaks`` path in
    :func:`_downbeat_times` — ~1 onset per real kick.
    """
    if env.size < 3:
        return np.empty(0, dtype=int)
    interior = np.arange(1, env.size - 1)
    is_max = (env[interior] > env[interior - 1]) & (
        env[interior] >= env[interior + 1]
    )
    cand = interior[is_max & (env[interior] >= threshold)]
    if cand.size == 0:
        return np.empty(0, dtype=int)
    order = cand[np.argsort(env[cand])[::-1]]  # strongest first
    kept: list[int] = []
    for peak in order:
        if all(abs(int(peak) - k) >= min_gap_frames for k in kept):
            kept.append(int(peak))
    return np.sort(np.asarray(kept, dtype=int))


def _downbeat_times(
    env: np.ndarray, sr: int, bpm: float, threshold: float
) -> np.ndarray:
    """One onset time per real kick from a low-band flux envelope.

    Peak-picks with ``scipy.signal.find_peaks`` (``height`` = the
    strength bar, ``distance`` = 0.6 of a beat in frames), falling back
    to the greedy numpy picker above when scipy is unavailable. The
    minimum distance is the fix for D1 round 3: ``librosa.onset_detect``
    with defaults returns every flux local maximum — on a dense drop
    (loud bed + 55 Hz bass) each kick's 150->50 Hz sweep beats against
    the steady tones in shared bins and yields several maxima per kick
    (222 onsets for 80 real kicks on the verifier's dense master), so
    the median offset was 66 ms on a perfectly on-grid track. With a
    0.6-beat minimum distance the same envelope yields exactly 80
    onsets, median offset +2.1 ms. Real kicks are at most one per beat,
    so a 0.6-beat gap can never merge two real kicks; it only kills the
    spurious maxima clustered around each kick.

    The first two frames are nulled first: a track starting from digital
    silence makes frame 1 the largest flux in the track (frame 0 is
    ~0), which would otherwise anchor the grid ~6 ms in and read every
    kick late. A real kick at t=0 still shows up in frames 2+; the
    constant ~6-12 ms shift cancels in every relative offset.
    """
    env = np.asarray(env, dtype=float).copy()
    env[:2] = 0.0
    beat_interval = 60.0 / float(bpm)
    min_gap_frames = max(
        1, int(round(0.6 * beat_interval * sr / _ONSET_HOP_LENGTH))
    )
    try:
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(env, height=threshold, distance=min_gap_frames)
    except ImportError:
        peaks = _pick_peaks_numpy(env, threshold, min_gap_frames)
    return peaks.astype(float) * _ONSET_HOP_LENGTH / sr


def _beat_drift(
    audio_2d: np.ndarray, sr: int, bpm: float | None
) -> tuple[list[str], float | None, list[str]]:
    """Median downbeat-to-beat offset vs the bpm grid (issue if > 30 ms).

    Returns ``(issues, signed_median_offset_ms, notes)``.

    Only low-frequency (kick/downbeat) onsets are timing evidence: the
    onset envelope is the positive spectral flux of the <= 200 Hz STFT
    bins, so off-beat hats, syncopated percussion and steady pads cannot
    phase the median. The old metric took the median offset of *all*
    strong onsets against a 0-anchored grid, which false-rejected every
    normal track (125-178 ms medians on in-time 120 BPM masters). The
    round-3 fix is in the peak-picking itself: ``scipy.find_peaks``
    with a 0.6-beat minimum distance (see :func:`_downbeat_times`),
    so each kick contributes exactly one onset even on a dense drop —
    ``librosa.onset_detect`` defaults returned several flux maxima per
    kick there and false-rejected on-grid masters at 66 ms median.

    The beat grid's phase is the circular mean of ALL detected
    downbeats' phases, not the first detection alone (D12): the flux
    peak of the first downbeat lags a slow attack by 30-50 ms (measured:
    a 50 ms fade-in at the track start puts the first detection at
    0.0406 s for a true first kick at 0.0 s, and the median offset reads
    44 ms on a perfectly on-grid track), while the rest of the train is
    detected accurately — so anchoring the grid at one detection
    false-rejects on-grid tracks and the loop "fixes" it with a spurious
    beat_shift_ms. One anomalous detection among ~80 cannot move the
    circular mean by more than ~1 ms, so the phase is the pulse train's
    own phase. This is equivalent to the old first-detection anchor
    whenever the first detection is accurate (both recover the train's
    phase), and it preserves the property that a track whose first kick
    lands at 70 ms is in time, not 70 ms late — the grid is the train's
    phase, not sample 0.

    The producer-style forward RMS refinement was considered and does
    NOT fix this case: it scans forward from the flux candidate for the
    first 50%-of-max crossing, which only corrects an EARLY flux peak
    (the attack still ahead). Here the flux peak is LATE — the attack
    is already past and the envelope at the candidate is above the bar —
    so it returns the candidate almost unchanged (measured: 0.0406 s ->
    0.0429 s, residual lag 42.9 ms). A backward RMS walk is
    threshold-fragile on the master: the 55 Hz bed keeps the 1 ms RMS
    floor high and oscillating at 110 Hz, so no fixed fraction of the
    local max reliably marks the attack start.

    The issue decision uses the *absolute* median of the wrapped
    offsets: on a tempo-mismatched track the phase drifts through the
    full cycle, so the absolute median stays large while a signed
    median would wash out to ~0. The *signed* median is returned
    alongside it as the constant phase of the pulse train, which the
    Engineer corrects with ``beat_shift_ms``.

    A flat flux envelope means there are no downbeats to measure —
    steady pads / drones — and the check is skipped with a note, never
    an issue. A missing librosa install or an onset-detection error
    also skips the check with a note.

    The skip is decided by the peak-picker itself, not by a
    max-vs-median ratio (D10): the old rule skipped whenever
    ``max <= 10x median``, but on a full real song the low band is
    continuously busy, so the median flux is high (18.23 on the
    "Veezha Veera" master, max 176.11 — only 9.66x) and the check
    skipped a track full of downbeats. Now the check runs the picker
    first and skips only when fewer than 4 strong onsets are found.
    That still skips a flat pad cleanly — there the envelope is nearly
    constant, so the strength bar (5x the median, and the median is
    ~the max) sits above every frame and zero peaks are found — while
    a busy low band still yields strong onsets above the bar, so the
    check runs and judges the pulse train.
    """
    if bpm is None or bpm <= 0:
        return [], None, []
    try:
        import librosa
    except ImportError:
        return [], None, ["Beat-drift check skipped — librosa is not installed"]
    mono = audio_2d.mean(axis=0)
    try:
        env = _low_band_flux(mono, sr)
    except Exception:  # noqa: BLE001 - onset detection must not kill the critic
        return [], None, ["Beat-drift check failed — onset detection errored"]
    if env.size == 0:
        return [], None, ["Beat-drift check skipped — no downbeat onsets detected"]
    # No max-vs-median pre-filter here (D10): a busy low band on a real
    # song has a high median flux, so a max/median ratio test misfires
    # and skips tracks full of downbeats. The skip is decided below by
    # the peak-picker — fewer than 4 strong onsets means no measurable
    # pulse train. Silence is the only envelope-level skip.
    env_max = float(np.max(env))
    if env_max <= 0.0:
        return [], None, ["Beat-drift check skipped — no downbeat onsets detected"]
    # Peak-pick the flux envelope with a 0.6-beat minimum distance, so
    # each kick contributes exactly one onset. (librosa.onset_detect
    # with defaults was the D1 bug: no minimum distance and a delta that
    # is meaningless on an unnormalized envelope, returning 222 onsets
    # for 80 real kicks on a dense drop.) Keep only strong onsets:
    # groove-defining kicks sit well above the noise-floor flux. The
    # bar is anchored to the floor (a multiple of the median frame flux)
    # and a small fraction of the peak — NOT to half the peak: a kick's
    # flux varies ~15x with its alignment to the STFT frame grid, so a
    # 50%-of-max bar keeps only the few best-aligned onsets (which
    # happen to fall exactly on the grid) and blinds the metric.
    threshold = max(5.0 * float(np.median(env)), 0.05 * float(np.max(env)))
    try:
        strong = _downbeat_times(env, sr, float(bpm), threshold)
    except Exception:  # noqa: BLE001 - onset detection must not kill the critic
        return [], None, ["Beat-drift check failed — onset detection errored"]
    if len(strong) == 0:
        return [], None, ["Beat-drift check skipped — no downbeat onsets detected"]
    if len(strong) < 4:
        return [], None, [
            f"Beat-drift check skipped — only {len(strong)} strong downbeat "
            "onset(s) detected (need >= 4)"
        ]
    beat_interval = 60.0 / float(bpm)
    # D12: grid phase from the circular mean of all strong onsets'
    # phases (mod one beat), not from the first detection alone — see
    # the docstring above. Equivalent to the old anchor when the first
    # detection is accurate; robust when it lags a slow first attack.
    phases = np.asarray(strong, dtype=float) % beat_interval
    angles = 2.0 * np.pi * phases / beat_interval
    grid_phase = (
        math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))
        % (2.0 * math.pi)
    ) * beat_interval / (2.0 * math.pi)
    # Signed offset of each onset to its nearest grid point, in
    # [-beat_interval/2, +beat_interval/2]; absolute offset is |signed|.
    signed = (
        (np.asarray(strong, dtype=float) - grid_phase + beat_interval / 2.0)
        % beat_interval
    ) - beat_interval / 2.0
    absolute = np.abs(signed)
    median_ms = float(np.median(absolute)) * 1000.0
    signed_median_ms = float(np.median(signed)) * 1000.0
    if median_ms > BEAT_OFFSET_LIMIT_MS:
        # A near-zero signed median with a large absolute median means
        # the offset is NOT a constant phase: the pulse train drifts
        # (tempo error) or two pulse trains are interleaved (e.g. source
        # kicks plus a synthesized kick pattern on a different grid).
        # Either way a whole-mix shift alone cannot fix it — say so,
        # rather than misdiagnosing one cause or the other.
        not_constant_phase = abs(signed_median_ms) < 5.0
        detail = (
            " — the offset varies across the track, so this is not a "
            "constant phase shift (a whole-mix shift alone cannot fix it)"
            if not_constant_phase
            else ""
        )
        return [
            f"Beat-sync drift: median downbeat-to-beat offset {median_ms:.0f} ms "
            f"> {BEAT_OFFSET_LIMIT_MS:.0f} ms at {bpm:g} BPM "
            f"(grid phase {grid_phase:.2f} s, circular mean of "
            f"{len(strong)} downbeats){detail}"
        ], signed_median_ms, []
    return [], signed_median_ms, []


def _spectral_balance(audio_2d: np.ndarray, sr: int) -> dict[str, float]:
    """Mean per-band level in dBFS, averaged over 1 s windows."""
    mono = audio_2d.mean(axis=0)
    win = sr  # 1 second
    if mono.shape[0] < win:
        win = mono.shape[0]
    freqs = np.fft.rfftfreq(win, d=1.0 / sr)
    spectra: list[np.ndarray] = []
    for start in range(0, mono.shape[0] - win + 1, win):
        spectra.append(np.abs(np.fft.rfft(mono[start : start + win])) ** 2)
    if not spectra:
        return {name: -120.0 for name, _, _ in _SPECTRAL_BANDS}
    mean_power = np.mean(np.stack(spectra), axis=0)
    balance: dict[str, float] = {}
    for name, lo, hi in _SPECTRAL_BANDS:
        band = (freqs >= lo) & (freqs < hi)
        power = float(np.mean(mean_power[band])) if np.any(band) else 0.0
        balance[name] = round(10.0 * math.log10(power + _EPS), 2)
    return balance


# ---------------------------------------------------------------------------
# Corrective parameters for the re-render
# ---------------------------------------------------------------------------


def _derive_fixes(
    issues: list[str],
    mix_params: dict[str, Any],
    lufs_delta: float,
    target_lufs: float,
    beat_offset_ms: float | None = None,
    transitions: list[tuple[str, str, float]] | None = None,
) -> dict[str, Any]:
    """Concrete DSP corrections, keyed for the Engineer agent.

    Only the parameters relevant to the issues found are set; everything
    else is left for the Engineer to keep from the previous render.
    Every key derived here is one the Engineer actually reads
    (``target_lufs``, ``limiter_ceiling_db``, ``bass_gain_db``,
    ``sidechain_depth_db``, ``section_gains``, ``beat_shift_ms``) — a dead
    knob is worse than no knob, because the loop re-renders identical
    audio while believing it corrected something. ``gain_trim_db`` was
    dropped for exactly that reason: the Engineer normalizes to
    ``target_lufs`` anyway, so a pre-normalization trim is normalized
    away. ``crossfade_ms`` was dropped the same way (D9): a longer
    crossfade cannot change a bar-window RMS step, so a transition
    rejection re-rendered the identical fix 3 times and never converged.
    A transition is corrected with ``section_gains`` instead — a trim on
    the louder section, which the Engineer applies before the mix bus.
    """
    fixes: dict[str, Any] = {}
    joined = " ".join(issues)
    if "Loudness" in joined:
        if lufs_delta < 0:
            # Ceiling-limited: the Engineer could not reach the target —
            # the true-peak ceiling forced the render quieter than asked.
            # The old formula aimed at ``target - delta`` (+5.73 LUFS
            # for a -7.73 LUFS render against a -1.0 target), which moves
            # the target *away* from the achievable level and doubles
            # the error on the next render. Aim AT the level the
            # Engineer actually achieved (``target + delta == measured``):
            # it is reachable, so the next render lands on target and
            # the error closes instead of diverging.
            revised = target_lufs + lufs_delta
        else:
            # Too loud: aim lower by exactly the overshoot, on the
            # assumption the overshoot is systematic.
            revised = target_lufs - lufs_delta
        fixes["target_lufs"] = round(min(max(revised, -30.0), 0.0), 2)
    if "True peak" in joined or "Clipping" in joined:
        fixes["limiter_ceiling_db"] = -1.5
    if "Clipping" in joined:
        fixes["bass_gain_db"] = round(float(mix_params.get("bass_gain_db", 0.0)) - 3.0, 2)
    if "masking" in joined.lower():
        fixes["sidechain_depth_db"] = round(
            float(mix_params.get("sidechain_depth_db", 6.0)) + 2.0, 2
        )
    if transitions:
        # Trim the louder section of each flagged boundary so the
        # residual step lands at ~6 dB (half the limit). The trim is a
        # scalar on that section and the Engineer's LUFS normalization
        # afterwards is a global scalar, so the relative step closes by
        # exactly the trim amount and the next render converges — unlike
        # the old crossfade_ms fix, which a bar-window RMS step cannot
        # see at all. Gains multiply onto any existing section_gains so
        # a second iteration trims further instead of overwriting.
        gains = dict(mix_params.get("section_gains") or {})
        for _text, louder_section, jump_db in transitions:
            trim_db = -(abs(jump_db) - TRANSITION_TARGET_DB)
            gains[louder_section] = round(
                float(gains.get(louder_section, 1.0))
                * 10.0 ** (trim_db / 20.0),
                4,
            )
        fixes["section_gains"] = gains
    if "Beat-sync drift" in joined and beat_offset_ms is not None:
        # Shift the whole mix so the downbeat pulse lands on the grid.
        # Signed median offset: positive means the pulse is late, so
        # the correction is its negative.
        shift = round(float(np.clip(-beat_offset_ms, -250.0, 250.0)), 2)
        fixes["beat_shift_ms"] = shift + 0.0  # normalize -0.0 to 0.0
    return fixes


# ---------------------------------------------------------------------------
# The critic node
# ---------------------------------------------------------------------------


def critic_node(
    state: RemixAgentState,
    llm: Callable[[dict[str, Any]], str] | None = None,
) -> dict:
    """Evaluate the master at ``state["master_path"]`` and return a verdict.

    Returns a LangGraph state update with ``critic_report``, ``approved``
    and, on a correctable rejection, an incremented ``iteration_count``
    plus merged corrective ``mix_params``. Non-fatal diagnostics (a
    skipped check, a stereo-width warning, an LLM note) are returned in
    the ``notes`` channel — never in ``errors`` and never as blocking
    issues, so a clean master with a skipped check still approves. The
    metric verdict is complete without an LLM; when ``llm`` is given,
    its text is appended to the notes and never overrides the metrics.
    """
    master_path = state.get("master_path")
    if not master_path:
        raise ValueError("critic_node needs state['master_path'] — no master was rendered")
    audio_2d, sr = _load_master(master_path)
    if sr != SAMPLE_RATE:
        raise ValueError(
            f"master sample rate is {sr} Hz, studio-wide rate is {SAMPLE_RATE} Hz"
        )

    mix_params = dict(state.get("mix_params") or {})
    target_lufs = float(mix_params.get("target_lufs", DEFAULT_TARGET_LUFS))
    iteration = int(state.get("iteration_count", 0))
    issues: list[str] = []
    notes: list[str] = []

    # (a) Loudness / true peak / clipping.
    loudness = measure_loudness(audio_2d, sr)
    lufs = float(loudness["integrated_lufs"])
    true_peak = float(loudness["true_peak_dbtp"])
    if not math.isfinite(lufs):
        issues.append("Master is silent — integrated loudness is -inf LUFS")
        lufs = -120.0
    if not math.isfinite(true_peak):
        true_peak = -120.0
    lufs_delta = lufs - target_lufs
    if abs(lufs_delta) > LUFS_TOLERANCE_DB:
        issues.append(
            f"Loudness {lufs:.2f} LUFS is {lufs_delta:+.2f} dB from target "
            f"{target_lufs:.1f} LUFS (tolerance +/-{LUFS_TOLERANCE_DB:.1f} dB)"
        )
    if true_peak > TRUE_PEAK_CEILING_DBTP:
        issues.append(
            f"True peak {true_peak:.2f} dBTP exceeds "
            f"{TRUE_PEAK_CEILING_DBTP:.1f} dBTP ceiling"
        )
    clip = detect_clipping(audio_2d)
    clipping = bool(loudness["clipping"]) or int(clip["clipped_samples"]) > 0
    if clipping:
        issues.append(
            f"Clipping detected: {int(clip['clipped_samples'])} clipped samples "
            f"({float(clip['percentage']):.3f}% of samples)"
        )

    # (b) Phase correlation.
    phase_corr, phase_warning = _phase_correlation(audio_2d)
    if phase_warning is not None:
        notes.append(phase_warning)
    if phase_corr < PHASE_CORRELATION_MIN:
        issues.append(
            f"Phase correlation {phase_corr:.2f} < {PHASE_CORRELATION_MIN:.2f} "
            "— stereo image collapsing / out of phase"
        )

    # (c) Transition roughness at blueprint boundaries (bar-window RMS,
    # phase-robust — see _transition_roughness). The structured results
    # carry the louder section + jump per boundary for the fix.
    transition_results = _transition_roughness(
        audio_2d, sr, state.get("blueprint"), bpm=state.get("bpm")
    )
    issues.extend(text for text, _louder, _jump in transition_results)

    # (d) Vocal masking estimate.
    issues.extend(_vocal_masking(audio_2d, sr, state.get("stems") or {}))

    # (e) Beat-sync drift (downbeat onsets only, grid phase is the
    # circular mean of all strong onsets' phases — D12). Skip/failure
    # messages are notes, never blocking issues.
    drift_issues, drift_offset_ms, drift_notes = _beat_drift(
        audio_2d, sr, state.get("bpm")
    )
    issues.extend(drift_issues)
    notes.extend(drift_notes)

    spectral_balance = _spectral_balance(audio_2d, sr)
    fixes = _derive_fixes(
        issues,
        mix_params,
        lufs_delta,
        target_lufs,
        beat_offset_ms=drift_offset_ms,
        transitions=transition_results,
    )

    approved = not issues
    verdict = "approved" if approved else "needs_revision"
    if approved:
        summary = (
            f"Master approved at iteration {iteration}: {lufs:.2f} LUFS "
            f"(target {target_lufs:.1f}), true peak {true_peak:.2f} dBTP, "
            f"phase correlation {phase_corr:.2f}, no blocking issues."
        )
    elif iteration >= MAX_CRITIC_ITERATIONS:
        summary = (
            f"Master rejected at iteration {iteration} with {len(issues)} issue(s), "
            "and the re-render budget is exhausted — shipping the best "
            "available render is a human decision now."
        )
    else:
        summary = (
            f"Master rejected at iteration {iteration} with {len(issues)} issue(s); "
            f"re-rendering with corrective parameters ({', '.join(sorted(fixes))})."
        )
    if llm is not None:
        try:
            llm_text = llm(
                {
                    "lufs": lufs,
                    "target_lufs": target_lufs,
                    "true_peak_db": true_peak,
                    "clipping": clipping,
                    "phase_correlation": phase_corr,
                    "issues": list(issues),
                    "spectral_balance": dict(spectral_balance),
                }
            )
            notes.append(f"LLM note: {llm_text}")
        except Exception as exc:  # noqa: BLE001 - LLM is advisory only
            notes.append(f"LLM verdict unavailable: {exc}")

    report = CriticReport(
        iteration=iteration,
        lufs_integrated=round(lufs, 2),
        true_peak_db=round(true_peak, 2),
        clipping=clipping,
        phase_correlation=round(phase_corr, 3),
        verdict=verdict,
        spectral_balance=spectral_balance,
        issues=issues,
        suggested_fixes=fixes,
        summary=summary,
    )

    update: dict[str, Any] = {
        "critic_report": report,
        "approved": approved,
        "notes": notes,
    }
    if not approved and iteration < MAX_CRITIC_ITERATIONS:
        merged = dict(mix_params)
        merged.update(fixes)
        update["iteration_count"] = iteration + 1
        update["mix_params"] = merged
    return update


def make_critic_node(
    llm: Callable[[dict[str, Any]], str] | None = None,
) -> Callable[[RemixAgentState], dict]:
    """Build a single-argument LangGraph node with an optional LLM hook."""

    def _node(state: RemixAgentState) -> dict:
        return critic_node(state, llm=llm)

    return _node


def route_after_critic(state: RemixAgentState) -> str:
    """``"re_render"`` while a rejected master still has budget, else ``"done"``.

    The state seen here is post-``critic_node`` (``iteration_count`` was
    already incremented on a correctable rejection), so a rejection at
    iteration 2 lands on 3 and routes to ``"done"`` — at most 3 renders.
    """
    if state.get("approved"):
        return "done"
    if state.get("critic_report") is None:
        return "done"
    if int(state.get("iteration_count", 0)) < MAX_CRITIC_ITERATIONS:
        return "re_render"
    return "done"
