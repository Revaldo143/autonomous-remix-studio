"""core/dsp_utils.py — Sample-accurate DSP primitives for the
Autonomous Multi-Agent AI Music Producer & Remix Studio.

Every function in this module operates on exact sample indices at a fixed
sample rate (default 44100 Hz).  Audio is represented as ``numpy`` arrays
with shape ``(samples,)`` for mono or ``(channels, samples)`` for
multi-channel audio.  No function ever estimates a position in seconds and
then rounds ad hoc: all slicing, stretching and envelope work is computed
from integer sample positions, and every length-critical output is trimmed
or zero-padded to its exact target length in samples.

Optional backends
-----------------
* ``pyrubberband`` — used first for time-stretching when installed.
* ``librosa`` — used next (``librosa.effects.time_stretch``).
* Pure NumPy/SciPy fallback — a from-scratch phase-vocoder time-stretcher
  built on ``scipy.signal.stft``/``istft``.  This is a real implementation,
  not a stub: it is what runs when neither backend is installed, and it is
  exercised by the ``__main__`` self-test.

Loudness is measured with a from-scratch BS.1770-4 implementation
(K-weighting via RBJ-cookbook biquads, 400 ms blocks with 75 % overlap,
absolute + relative gating).  ``pyloudnorm`` is never required.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal as _signal

__all__ = [
    "ms_to_samples",
    "samples_to_ms",
    "align_to_beat_grid",
    "kick_triggered_sidechain",
    "dynamic_vocal_carve",
    "time_stretch_formant_preserving",
    "binaural_pan",
    "measure_loudness",
    "detect_clipping",
    "normalize_to_lufs",
]

DEFAULT_SR = 44100

# ---------------------------------------------------------------------------
# Optional backends (imported lazily-tolerant: absence only selects a fallback)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - environment dependent
    import librosa as _librosa

    _HAS_LIBROSA = True
except Exception:  # noqa: BLE001 - any import failure means "not available"
    _librosa = None
    _HAS_LIBROSA = False

try:  # pragma: no cover - environment dependent
    import pyrubberband as _pyrubberband

    _HAS_RUBBERBAND = True
except Exception:  # noqa: BLE001 - any import failure means "not available"
    _pyrubberband = None
    _HAS_RUBBERBAND = False


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _to_2d(audio: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return ``(array_2d, was_1d)`` with shape ``(channels, samples)``."""
    a = np.asarray(audio)
    if a.ndim == 1:
        return a[np.newaxis, :].astype(np.float64, copy=False), True
    if a.ndim == 2:
        return a.astype(np.float64, copy=False), False
    raise ValueError(f"audio must be 1-D (samples,) or 2-D (channels, samples); got shape {a.shape}")


def _from_2d(array_2d: np.ndarray, was_1d: bool) -> np.ndarray:
    """Undo :func:`_to_2d` (mono input returns a 1-D array)."""
    if was_1d:
        return array_2d[0]
    return array_2d


def _fit_exact_length(x: np.ndarray, target_len: int) -> np.ndarray:
    """Trim or zero-pad a 1-D array to exactly ``target_len`` samples."""
    if target_len < 0:
        raise ValueError(f"target_len must be >= 0, got {target_len}")
    n = x.shape[0]
    if n == target_len:
        return x
    if n > target_len:
        return x[:target_len]
    out = np.zeros(target_len, dtype=x.dtype)
    out[:n] = x
    return out


# ---------------------------------------------------------------------------
# Time conversion (exact sample indices)
# ---------------------------------------------------------------------------
def ms_to_samples(ms: int, sr: int = DEFAULT_SR) -> int:
    """Convert milliseconds to an exact integer sample count (rounded)."""
    if sr <= 0:
        raise ValueError(f"sr must be > 0, got {sr}")
    return int(round(float(ms) * sr / 1000.0))


def samples_to_ms(samples: int, sr: int = DEFAULT_SR) -> int:
    """Convert an integer sample count to milliseconds (rounded)."""
    if sr <= 0:
        raise ValueError(f"sr must be > 0, got {sr}")
    return int(round(float(samples) * 1000.0 / sr))


# ---------------------------------------------------------------------------
# Time-stretching backends
# ---------------------------------------------------------------------------
def _phase_vocoder_stretch(
    x: np.ndarray, rate: float, n_fft: int = 2048, hop: int = 512
) -> np.ndarray:
    """Pure NumPy/SciPy phase-vocoder time-stretch (Laroche/Dolson).

    ``rate > 1`` speeds up (output shorter), ``rate < 1`` slows down.
    Magnitude is linearly interpolated between input STFT frames at the
    fractional read position while phase is advanced by the measured true
    frequency per bin, which keeps transients and pitch coherent.  Used as
    the fallback when neither pyrubberband nor librosa is installed.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n == 0 or rate == 1.0:
        return x.copy()
    if n < 64:
        # Degenerate path: too few samples for an STFT — plain linear
        # resampling keeps the function total (documented, not a stub).
        target = max(1, int(round(n / rate)))
        return np.interp(np.linspace(0, n - 1, target), np.arange(n), x)
    n_fft = min(n_fft, 1 << int(math.floor(math.log2(n))))
    hop = min(hop, n_fft // 4)

    _, _, z = _signal.stft(
        x, fs=1.0, window="hann", nperseg=n_fft, noverlap=n_fft - hop,
        boundary="zeros", padded=True,
    )
    n_bins, n_frames = z.shape
    mag = np.abs(z)
    phase = np.angle(z)
    # Expected phase advance per STFT hop, per frequency bin.
    expected = (2.0 * np.pi * hop * np.arange(n_bins) / n_fft)[:, np.newaxis]

    out_frames = int(math.ceil(n_frames / rate))
    y = np.zeros((n_bins, out_frames), dtype=np.complex128)
    phase_acc = phase[:, 0].copy()
    for out_idx in range(out_frames):
        pos = out_idx * rate
        idx = int(math.floor(pos))
        frac = pos - idx
        idx = min(idx, n_frames - 1)
        idx_next = min(idx + 1, n_frames - 1)
        m = (1.0 - frac) * mag[:, idx] + frac * mag[:, idx_next]
        # True frequency: expected advance plus wrapped phase deviation.
        dev = phase[:, idx_next] - phase[:, idx] - expected[:, 0]
        dev -= 2.0 * np.pi * np.round(dev / (2.0 * np.pi))
        phase_acc += expected[:, 0] + dev
        y[:, out_idx] = m * np.exp(1j * phase_acc)

    _, stretched = _signal.istft(
        y, fs=1.0, window="hann", nperseg=n_fft, noverlap=n_fft - hop,
        boundary=True,
    )
    return _fit_exact_length(stretched, max(1, int(round(n / rate))))


def _time_stretch_1d(x: np.ndarray, rate: float, sr: int) -> np.ndarray:
    """Stretch one mono channel by ``rate`` (>1 = faster/shorter).

    Backend order: pyrubberband -> librosa -> pure NumPy phase vocoder.
    The output is always exactly ``round(len(x) / rate)`` samples.
    """
    x = np.asarray(x, dtype=np.float64)
    if rate <= 0:
        raise ValueError(f"rate must be > 0, got {rate}")
    if x.shape[0] == 0 or rate == 1.0:
        return x.copy()
    target_len = max(1, int(round(x.shape[0] / rate)))
    if _HAS_RUBBERBAND:
        stretched = _pyrubberband.time_stretch(x, sr, rate=float(rate))
        return _fit_exact_length(np.asarray(stretched, dtype=np.float64), target_len)
    if _HAS_LIBROSA:
        stretched = _librosa.effects.time_stretch(x, rate=float(rate))
        return _fit_exact_length(np.asarray(stretched, dtype=np.float64), target_len)
    return _phase_vocoder_stretch(x, rate)


# ---------------------------------------------------------------------------
# Beat-grid alignment
# ---------------------------------------------------------------------------
def align_to_beat_grid(
    audio: np.ndarray, sr: int, bpm: float, downbeat_samples: list[int]
) -> np.ndarray:
    """Align audio to a sample-accurate beat grid.

    The audio is sliced at the given downbeat sample indices.  Every bar
    between consecutive downbeats is time-stretched to exactly one bar
    length (``4 beats * 60 / bpm`` seconds, in samples).  A leading partial
    segment (before the first downbeat) and a trailing partial segment
    (after the last downbeat) are stretched to the nearest whole number of
    bars (at least one).  After stretching, each segment is trimmed or
    zero-padded to its exact target length, so every bar boundary in the
    output lands exactly on the grid — no accumulated drift.

    Parameters
    ----------
    audio : np.ndarray
        Mono ``(samples,)`` or ``(channels, samples)``.
    sr : int
        Sample rate in Hz.
    bpm : float
        Tempo in beats per minute (must be > 0).
    downbeat_samples : list[int]
        Sample indices of bar starts (downbeats), in ascending order.

    Returns
    -------
    np.ndarray
        Same channel layout as the input; total length is an exact whole
        number of bars in samples.
    """
    if sr <= 0:
        raise ValueError(f"sr must be > 0, got {sr}")
    if bpm <= 0:
        raise ValueError(f"bpm must be > 0, got {bpm}")
    a2d, was_1d = _to_2d(audio)
    n_ch, n = a2d.shape
    bar_len = int(round(4.0 * 60.0 / bpm * sr))
    if bar_len < 1:
        raise ValueError(f"bpm {bpm} at sr {sr} gives a bar shorter than 1 sample")

    # Clean, sorted, in-range downbeat indices.
    beats = sorted({int(b) for b in downbeat_samples if 0 < int(b) < n})
    boundaries = [0] + beats + [n]

    segments: list[np.ndarray] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        seg_len = end - start
        if seg_len <= 0:
            continue
        is_full_bar = start in beats or (start == 0 and end in beats)
        if is_full_bar:
            target = bar_len
        else:
            # Leading / trailing partial: nearest whole number of bars.
            target = max(bar_len, int(round(seg_len / bar_len)) * bar_len)
        rate = seg_len / target
        stretched_ch = [
            _time_stretch_1d(a2d[ch, start:end], rate, sr) for ch in range(n_ch)
        ]
        seg = np.stack(
            [_fit_exact_length(s, target) for s in stretched_ch], axis=0
        )
        segments.append(seg)

    if not segments:
        return _from_2d(a2d.copy(), was_1d)
    out = np.concatenate(segments, axis=1)
    return _from_2d(out, was_1d)


# ---------------------------------------------------------------------------
# Kick-triggered sidechain compression (envelope follower)
# ---------------------------------------------------------------------------
def kick_triggered_sidechain(
    audio: np.ndarray,
    kick_times_samples: list[int],
    sr: int,
    *,
    depth_db: float = 6.0,
    attack_ms: float = 5.0,
    release_ms: float = 120.0,
    lookahead_ms: float = 2.0,
) -> np.ndarray:
    """Duck ``audio`` on every kick trigger, sample-accurately.

    A gain envelope is built from the kick trigger sample positions:
    starting ``lookahead_ms`` before each kick the gain dives from 1.0 to
    the ducked level (``10 ** (-depth_db / 20)``) over ``attack_ms`` with a
    raised-cosine ramp, then returns to 1.0 over ``release_ms`` with a
    matching raised-cosine ramp.  Overlapping kicks combine by taking the
    deepest (minimum) gain at each sample, so the envelope is click-free
    by construction — no discontinuities anywhere in the gain curve.

    Parameters
    ----------
    audio : np.ndarray
        Mono ``(samples,)`` or ``(channels, samples)``.
    kick_times_samples : list[int]
        Sample indices of kick onsets.
    sr : int
        Sample rate in Hz.
    depth_db : float
        Maximum gain reduction in dB (must be >= 0).
    attack_ms, release_ms, lookahead_ms : float
        Envelope timing in milliseconds (must be >= 0).

    Returns
    -------
    np.ndarray
        Same shape as the input (float64 for integer input).
    """
    if sr <= 0:
        raise ValueError(f"sr must be > 0, got {sr}")
    if depth_db < 0:
        raise ValueError(f"depth_db must be >= 0, got {depth_db}")
    for name, val in (
        ("attack_ms", attack_ms),
        ("release_ms", release_ms),
        ("lookahead_ms", lookahead_ms),
    ):
        if val < 0:
            raise ValueError(f"{name} must be >= 0, got {val}")

    a2d, was_1d = _to_2d(audio)
    n_ch, n = a2d.shape
    if n == 0:
        return _from_2d(a2d.copy(), was_1d)

    attack = max(1, ms_to_samples(int(round(attack_ms)), sr))
    release = max(1, ms_to_samples(int(round(release_ms)), sr))
    lookahead = ms_to_samples(int(round(lookahead_ms)), sr)
    min_gain = 10.0 ** (-depth_db / 20.0)

    envelope = np.ones(n, dtype=np.float64)
    for kick in kick_times_samples:
        start = int(kick) - lookahead
        if start + release < 0 or start >= n:
            continue
        # Attack ramp: 1.0 -> min_gain (raised cosine).
        a0, a1 = max(start, 0), min(start + attack, n)
        if a1 > a0:
            t = np.arange(a0, a1, dtype=np.float64) - start
            ramp = 1.0 - (1.0 - min_gain) * 0.5 * (
                1.0 - np.cos(np.pi * t / attack)
            )
            envelope[a0:a1] = np.minimum(envelope[a0:a1], ramp)
        # Hold at min_gain for the remainder of the attack window is
        # implicit: the release ramp starts at min_gain.
        # Release ramp: min_gain -> 1.0 (raised cosine).
        r0, r1 = max(start + attack, 0), min(start + attack + release, n)
        if r1 > r0:
            t = np.arange(r0, r1, dtype=np.float64) - (start + attack)
            ramp = min_gain + (1.0 - min_gain) * 0.5 * (
                1.0 - np.cos(np.pi * t / release)
            )
            envelope[r0:r1] = np.minimum(envelope[r0:r1], ramp)

    out = a2d * envelope[np.newaxis, :]
    if np.issubdtype(np.asarray(audio).dtype, np.floating):
        out = out.astype(np.asarray(audio).dtype, copy=False)
    return _from_2d(out, was_1d)


# ---------------------------------------------------------------------------
# Dynamic spectral vocal carving
# ---------------------------------------------------------------------------
def dynamic_vocal_carve(
    instrumental: np.ndarray,
    vocal: np.ndarray,
    sr: int,
    *,
    bands: int = 8,
    reduction_db: float = 3.0,
) -> np.ndarray:
    """Carve vocal frequencies out of an instrumental stem, dynamically.

    Both signals are transformed with an STFT.  Frequency bins are grouped
    into ``bands`` log-spaced bands.  In each band and frame the vocal
    dominance ``d = E_vocal / (E_vocal + E_inst)`` is measured; the
    instrumental magnitude in that band is attenuated by up to
    ``reduction_db`` in proportion to dominance above a 0.35 threshold.
    The per-band attenuation is smoothed over time with an asymmetric
    one-pole filter (fast attack, slow release) so the carving tracks the
    vocal without pumping.  Phase is preserved and the result is ISTFT'd
    back to the time domain.

    Parameters
    ----------
    instrumental, vocal : np.ndarray
        Mono ``(samples,)`` or ``(channels, samples)``.  The vocal is
        analysed per channel; if it has fewer channels than the
        instrumental its single channel is reused for all of them.
    sr : int
        Sample rate in Hz.
    bands : int
        Number of log-spaced frequency bands (must be >= 1).
    reduction_db : float
        Maximum attenuation in dB (must be >= 0).

    Returns
    -------
    np.ndarray
        Same shape and dtype as ``instrumental``.
    """
    if sr <= 0:
        raise ValueError(f"sr must be > 0, got {sr}")
    if bands < 1:
        raise ValueError(f"bands must be >= 1, got {bands}")
    if reduction_db < 0:
        raise ValueError(f"reduction_db must be >= 0, got {reduction_db}")

    inst_2d, inst_1d = _to_2d(instrumental)
    voc_2d, _ = _to_2d(vocal)
    n_ch, n = inst_2d.shape
    orig_dtype = np.asarray(instrumental).dtype
    if n == 0:
        return np.asarray(instrumental).copy()

    # One vocal channel per instrumental channel (broadcast a mono vocal).
    if voc_2d.shape[0] == 1 and n_ch > 1:
        voc_2d = np.repeat(voc_2d, n_ch, axis=0)
    elif voc_2d.shape[0] != n_ch:
        raise ValueError(
            f"vocal has {voc_2d.shape[0]} channel(s) but instrumental has {n_ch}"
        )

    n_fft = 2048
    hop = n_fft // 4
    eps = 1e-12
    out_ch = []
    for ch in range(n_ch):
        _, _, z_inst = _signal.stft(
            inst_2d[ch], fs=sr, window="hann", nperseg=n_fft,
            noverlap=n_fft - hop, boundary="zeros", padded=True,
        )
        _, _, z_voc = _signal.stft(
            voc_2d[ch], fs=sr, window="hann", nperseg=n_fft,
            noverlap=n_fft - hop, boundary="zeros", padded=True,
        )
        n_frames = min(z_inst.shape[1], z_voc.shape[1])
        z_inst, z_voc = z_inst[:, :n_frames], z_voc[:, :n_frames]
        mag_inst = np.abs(z_inst)
        mag_voc = np.abs(z_voc)

        # Log-spaced band edges over the STFT bins (20 Hz .. Nyquist).
        n_bins = mag_inst.shape[0]
        f_min, f_max = 20.0, sr / 2.0
        edges = np.unique(
            np.clip(
                np.round(
                    np.logspace(np.log10(f_min), np.log10(f_max), bands + 1)
                    / (sr / 2.0) * (n_bins - 1)
                ).astype(int),
                0, n_bins - 1,
            )
        )
        att_db = np.zeros((n_bins, n_frames), dtype=np.float64)
        for b in range(len(edges) - 1):
            lo, hi = edges[b], max(edges[b + 1], edges[b] + 1)
            e_voc = mag_voc[lo:hi].mean(axis=0)
            e_inst = mag_inst[lo:hi].mean(axis=0)
            dominance = e_voc / (e_voc + e_inst + eps)
            target = reduction_db * np.clip((dominance - 0.35) / 0.65, 0.0, 1.0)
            # Asymmetric smoothing: fast attack (0.5), slow release (0.05).
            smoothed = np.zeros_like(target)
            acc = 0.0
            for t in range(n_frames):
                alpha = 0.5 if target[t] > acc else 0.05
                acc = alpha * target[t] + (1.0 - alpha) * acc
                smoothed[t] = acc
            att_db[lo:hi, :] = smoothed[np.newaxis, :]

        gain = 10.0 ** (-att_db / 20.0)
        carved = z_inst * gain
        _, y = _signal.istft(
            carved, fs=sr, window="hann", nperseg=n_fft,
            noverlap=n_fft - hop, boundary=True,
        )
        out_ch.append(_fit_exact_length(np.asarray(y, dtype=np.float64), n))

    out = np.stack(out_ch, axis=0)
    out = _from_2d(out, inst_1d)
    if np.issubdtype(orig_dtype, np.floating):
        return out.astype(orig_dtype, copy=False)
    # Integer input: round back and clip to the original integer range.
    info = np.iinfo(orig_dtype)
    return np.clip(np.rint(out), info.min, info.max).astype(orig_dtype)


# ---------------------------------------------------------------------------
# Formant-preserving time-stretch
# ---------------------------------------------------------------------------
def time_stretch_formant_preserving(
    audio: np.ndarray, sr: int, rate: float
) -> np.ndarray:
    """Time-stretch audio while preserving formants (and therefore pitch).

    ``rate > 1`` speeds up (shorter), ``rate < 1`` slows down.  Backend
    order: pyrubberband (Rubber Band Library, inherently formant-safe for
    time-stretching) -> librosa phase vocoder -> pure NumPy phase vocoder.
    The output length is exactly ``round(len / rate)`` samples per channel.

    Raises
    ------
    ValueError
        If ``rate <= 0``.
    """
    if rate <= 0:
        raise ValueError(f"rate must be > 0, got {rate}")
    a2d, was_1d = _to_2d(audio)
    n_ch, n = a2d.shape
    target_len = max(1, int(round(n / rate)))
    out_ch = [
        _fit_exact_length(_time_stretch_1d(a2d[ch], rate, sr), target_len)
        for ch in range(n_ch)
    ]
    out = np.stack(out_ch, axis=0)
    if np.issubdtype(np.asarray(audio).dtype, np.floating):
        out = out.astype(np.asarray(audio).dtype, copy=False)
    return _from_2d(out, was_1d)


# ---------------------------------------------------------------------------
# Binaural panning
# ---------------------------------------------------------------------------
def binaural_pan(audio: np.ndarray, sr: int, azimuth_deg: float) -> np.ndarray:
    """Place audio in a stereo field at ``azimuth_deg`` (-90 .. +90).

    Combines a constant-power pan law with a simple interaural time
    difference (ITD): the ear opposite the source is delayed by up to
    ~0.6 ms (``0.6 ms * sin(azimuth)``), applied as a fractional delay
    via linear interpolation so the shift is sample-accurate at any
    sample rate.  Mono input becomes stereo ``(2, samples)``; stereo
    input is re-balanced with the same pan law and ITD.

    Parameters
    ----------
    audio : np.ndarray
        Mono ``(samples,)`` / ``(1, samples)`` or stereo ``(2, samples)``.
    sr : int
        Sample rate in Hz.
    azimuth_deg : float
        -90 = hard left, 0 = centre, +90 = hard right.

    Returns
    -------
    np.ndarray
        Stereo ``(2, samples)`` float64.
    """
    if sr <= 0:
        raise ValueError(f"sr must be > 0, got {sr}")
    if not -90.0 <= azimuth_deg <= 90.0:
        raise ValueError(f"azimuth_deg must be in [-90, 90], got {azimuth_deg}")

    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 1:
        mono = a
    elif a.ndim == 2 and a.shape[0] == 1:
        mono = a[0]
    elif a.ndim == 2 and a.shape[0] == 2:
        mono = None  # keep the two channels, balance them below
    else:
        raise ValueError(
            f"binaural_pan expects mono or stereo (2, samples); got shape {a.shape}"
        )

    # Constant-power pan law: azimuth -90 -> (1, 0), 0 -> (0.707, 0.707),
    # +90 -> (0, 1).
    pan_rad = math.radians((azimuth_deg + 90.0) / 2.0)
    g_left = math.cos(pan_rad)
    g_right = math.sin(pan_rad)

    if mono is not None:
        left = mono * g_left
        right = mono * g_right
    else:
        # Balance a stereo source with the same constant-power law: at
        # centre each channel is at 0.7071 (equal total power), at hard
        # left/right the far channel is muted and the near one is at 1.0.
        left = a[0] * g_left
        right = a[1] * g_right

    # Interaural time difference: delay the far ear by up to 0.6 ms.
    itd_s = 0.0006 * math.sin(math.radians(azimuth_deg))
    if itd_s > 0:
        right = _fractional_delay(right, 0.0)
        left = _fractional_delay(left, itd_s * sr)
    elif itd_s < 0:
        left = _fractional_delay(left, 0.0)
        right = _fractional_delay(right, -itd_s * sr)
    return np.stack([left, right], axis=0)


def _fractional_delay(x: np.ndarray, delay_samples: float) -> np.ndarray:
    """Delay a 1-D signal by a fractional number of samples.

    Linear interpolation between adjacent samples; the output keeps the
    input length (samples delayed past the end are dropped, the head is
    zero-filled).
    """
    n = x.shape[0]
    if n == 0 or delay_samples == 0.0:
        return x.copy()
    pos = np.arange(n, dtype=np.float64) - delay_samples
    return np.interp(pos, np.arange(n, dtype=np.float64), x, left=0.0, right=0.0)


# ---------------------------------------------------------------------------
# Loudness (BS.1770-4, from-scratch K-weighting + gating)
# ---------------------------------------------------------------------------
def _rbj_high_shelf(f0: float, gain_db: float, sr: int, slope: float = 1.0):
    """RBJ-cookbook high-shelf biquad coefficients ``(b, a)``.

    High-shelf form (Audio EQ Cookbook): the ``cos(w0)`` terms carry the
    opposite signs from the low shelf — ``+(A-1)*cos`` in ``b0``/``b2``
    and ``-(A-1)*cos`` in ``a0``/``a2``, with ``b1``/``a1`` built from
    ``(A-1) + (A+1)*cos`` / ``(A-1) - (A+1)*cos`` respectively.  (An
    earlier revision used the low-shelf signs here, which made the
    K-weighting pre-filter boost the lows by +4 dB instead of the highs.)
    With ``f0=1500 Hz, gain=+4 dB, slope=1`` this reproduces the
    BS.1770-4 48 kHz reference coefficients to ~1e-4.
    """
    a_amp = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / sr
    alpha = math.sin(w0) / 2.0 * math.sqrt(
        (a_amp + 1.0 / a_amp) * (1.0 / slope - 1.0) + 2.0
    )
    cos_w0 = math.cos(w0)
    sqrt_a = math.sqrt(a_amp)
    b = np.array([
        a_amp * ((a_amp + 1.0) + (a_amp - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha),
        -2.0 * a_amp * ((a_amp - 1.0) + (a_amp + 1.0) * cos_w0),
        a_amp * ((a_amp + 1.0) + (a_amp - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha),
    ])
    a = np.array([
        (a_amp + 1.0) - (a_amp - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha,
        2.0 * ((a_amp - 1.0) - (a_amp + 1.0) * cos_w0),
        (a_amp + 1.0) - (a_amp - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha,
    ])
    return b / a[0], a / a[0]


def _rbj_highpass(f0: float, q: float, sr: int):
    """RBJ-cookbook high-pass biquad coefficients ``(b, a)``."""
    w0 = 2.0 * math.pi * f0 / sr
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)
    b = np.array([(1.0 + cos_w0) / 2.0, -(1.0 + cos_w0), (1.0 + cos_w0) / 2.0])
    a = np.array([1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha])
    return b / a[0], a / a[0]


def _k_weight(audio_2d: np.ndarray, sr: int) -> np.ndarray:
    """Apply BS.1770 K-weighting (pre-filter shelf + RLB high-pass)."""
    # BS.1770-4 pre-filter: high shelf, f0 = 1500 Hz, +4 dB, slope 1.
    # (1500 Hz — not 1681.97 Hz — is the design frequency that reproduces
    # the standard's 48 kHz reference coefficients.)
    b_pre, a_pre = _rbj_high_shelf(1500.0, 4.0, sr)
    b_rlb, a_rlb = _rbj_highpass(38.0, 0.5, sr)
    out = np.empty_like(audio_2d)
    for ch in range(audio_2d.shape[0]):
        stage = _signal.lfilter(b_pre, a_pre, audio_2d[ch])
        out[ch] = _signal.lfilter(b_rlb, a_rlb, stage)
    return out


def _integrated_lufs(audio_2d: np.ndarray, sr: int) -> float:
    """BS.1770-4 gated integrated loudness (LUFS), -inf for silence."""
    n = audio_2d.shape[1]
    if n == 0:
        return float("-inf")
    weighted = _k_weight(audio_2d, sr)
    block = int(round(0.400 * sr))
    hop = int(round(0.100 * sr))
    if n < block:
        # Shorter than one 400 ms block: measure the whole signal as one.
        # BS.1770-4: z is the SUM over channels of per-channel mean
        # squares (G_i = 1.0 for L/R), never the mean across channels.
        z = float(np.sum(np.mean(weighted ** 2, axis=1)))
        return -0.691 + 10.0 * math.log10(z) if z > 0 else float("-inf")
    starts = range(0, n - block + 1, hop)
    z_blocks = np.array([
        float(np.sum(np.mean(weighted[:, s:s + block] ** 2, axis=1)))
        for s in starts
    ])
    loud_blocks = np.array([
        -0.691 + 10.0 * math.log10(z) if z > 0 else float("-inf")
        for z in z_blocks
    ])
    # Absolute gate: discard blocks at or below -70 LUFS.
    keep = loud_blocks > -70.0
    if not np.any(keep):
        return float("-inf")
    z_kept, loud_kept = z_blocks[keep], loud_blocks[keep]
    # Relative gate: discard blocks 10 LU below the mean of the rest.
    rel_threshold = float(np.mean(loud_kept)) - 10.0
    keep2 = loud_kept > rel_threshold
    if not np.any(keep2):
        return float("-inf")
    return -0.691 + 10.0 * math.log10(float(np.mean(z_kept[keep2])))


def _true_peak_dbtp(audio_2d: np.ndarray, oversample: int = 4) -> float:
    """True peak in dBTP via FFT-based oversampling (approximation)."""
    n = audio_2d.shape[1]
    if n == 0:
        return float("-inf")
    peak = 0.0
    for ch in range(audio_2d.shape[0]):
        up = _signal.resample(audio_2d[ch], n * oversample)
        peak = max(peak, float(np.max(np.abs(up))) if up.size else 0.0)
    return 20.0 * math.log10(peak) if peak > 0 else float("-inf")


def _db(value: float, floor: float = -120.0) -> float:
    """Linear magnitude to dB with a silence floor."""
    if value <= 0:
        return floor
    return max(20.0 * math.log10(value), floor)


def measure_loudness(audio: np.ndarray, sr: int) -> dict:
    """Measure loudness and peak metrics of ``audio``.

    Returns a dict with:

    ``integrated_lufs``
        BS.1770-4 gated integrated loudness (``-inf`` for digital silence).
    ``true_peak_dbtp``
        True peak in dBTP (4x oversampled approximation).
    ``peak_dbfs``
        Sample peak in dBFS.
    ``rms_db``
        RMS level in dB.
    ``clipping``
        ``True`` when any sample has ``|x| >= 0.999``.
    """
    if sr <= 0:
        raise ValueError(f"sr must be > 0, got {sr}")
    a2d, _ = _to_2d(audio)
    n = a2d.shape[1]
    if n == 0:
        return {
            "integrated_lufs": float("-inf"),
            "true_peak_dbtp": float("-inf"),
            "peak_dbfs": -120.0,
            "rms_db": -120.0,
            "clipping": False,
        }
    peak = float(np.max(np.abs(a2d)))
    rms = float(np.sqrt(np.mean(a2d ** 2)))
    return {
        "integrated_lufs": _integrated_lufs(a2d, sr),
        "true_peak_dbtp": _true_peak_dbtp(a2d),
        "peak_dbfs": _db(peak),
        "rms_db": _db(rms),
        "clipping": bool(np.any(np.abs(a2d) >= 0.999)),
    }


# ---------------------------------------------------------------------------
# Clipping detection
# ---------------------------------------------------------------------------
def detect_clipping(audio: np.ndarray, threshold: float = 0.999) -> dict:
    """Locate clipped samples.

    Returns a dict with ``clipped_samples`` (count across all channels),
    ``positions`` (sample indices of the first 100 clipped samples, in
    ascending order), ``percentage`` (of all samples) and ``threshold``.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    a2d, _ = _to_2d(audio)
    total = int(a2d.size)
    if total == 0:
        return {
            "clipped_samples": 0,
            "positions": [],
            "percentage": 0.0,
            "threshold": float(threshold),
        }
    # A sample index counts once even if several channels clip there.
    clipped_mask = np.any(np.abs(a2d) >= threshold, axis=0)
    positions = np.flatnonzero(clipped_mask)
    count = int(positions.size)
    return {
        "clipped_samples": count,
        "positions": [int(p) for p in positions[:100]],
        "percentage": 100.0 * count / int(a2d.shape[1]),
        "threshold": float(threshold),
    }


# ---------------------------------------------------------------------------
# LUFS normalization with a true-peak ceiling
# ---------------------------------------------------------------------------
def normalize_to_lufs(
    audio: np.ndarray, sr: int, target_lufs: float
) -> np.ndarray:
    """Gain ``audio`` to ``target_lufs`` integrated loudness.

    A single gain stage brings the BS.1770 integrated loudness to the
    target; afterwards the true peak is checked and, if it exceeds the
    hard ceiling of -1.0 dBTP, the signal is scaled down until it fits
    (up to 3 iterations).  In that case the final loudness sits below
    the target — the ceiling always wins over the target.  Digital
    silence is returned unchanged.
    """
    if sr <= 0:
        raise ValueError(f"sr must be > 0, got {sr}")
    a2d, was_1d = _to_2d(audio)
    if a2d.shape[1] == 0:
        return _from_2d(a2d.copy(), was_1d)
    measured = _integrated_lufs(a2d, sr)
    if not math.isfinite(measured):
        return _from_2d(a2d.copy(), was_1d)
    gain = 10.0 ** ((target_lufs - measured) / 20.0)
    out = a2d * gain
    # Aim 0.05 dB below the -1.0 dBTP ceiling: after resampling, a signal
    # scaled to exactly -1.0 dBTP can re-measure a hair above it due to
    # floating-point rounding, so the margin guarantees the ceiling holds.
    for _ in range(3):
        tp = _true_peak_dbtp(out)
        if tp <= -1.0:
            break
        out = out * 10.0 ** ((-1.05 - tp) / 20.0)
    if np.issubdtype(np.asarray(audio).dtype, np.floating):
        out = out.astype(np.asarray(audio).dtype, copy=False)
    return _from_2d(out, was_1d)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    SR = 44100
    DUR = 2.0
    N = int(SR * DUR)
    t = np.arange(N, dtype=np.float64) / SR

    # 2-second test tone: 440 Hz + 880 Hz, stereo with a slight right lift.
    tone_mono = 0.5 * np.sin(2 * np.pi * 440.0 * t) + 0.25 * np.sin(2 * np.pi * 880.0 * t)
    tone = np.stack([tone_mono, tone_mono * 0.9], axis=0)
    # Kick pattern: 120 BPM -> a kick every 0.5 s.
    kick_samples = [0, 22050, 44100, 66150]

    print("== dsp_utils self-test ==")
    print(f"sample rate: {SR} Hz | test tone: {N} samples ({DUR:.1f} s) stereo")

    print(f"[1] ms_to_samples(1000) = {ms_to_samples(1000, SR)} "
          f"(expect 44100) | samples_to_ms(44100) = {samples_to_ms(44100, SR)} "
          f"(expect 1000)")
    assert ms_to_samples(1000, SR) == 44100
    assert samples_to_ms(44100, SR) == 1000

    # --- Beat-grid alignment: 4 s tone, one off-grid downbeat at 80000. ---
    t4 = np.arange(4 * SR, dtype=np.float64) / SR
    tone4 = np.stack(
        [0.5 * np.sin(2 * np.pi * 440.0 * t4)] * 2, axis=0
    )
    aligned = align_to_beat_grid(tone4, SR, bpm=120.0, downbeat_samples=[80000])
    bar_len = int(round(4 * 60.0 / 120.0 * SR))
    print(f"[2] align_to_beat_grid: in {tone4.shape[1]} samples, "
          f"out {aligned.shape} | bar_len {bar_len} samples | "
          f"boundary at {bar_len} is exact: {aligned.shape[1] % bar_len == 0}")
    assert aligned.shape == (2, 4 * SR)
    assert aligned.shape[1] % bar_len == 0

    # --- Kick-triggered sidechain on a DC signal (envelope is readable). ---
    ones = np.ones((1, N), dtype=np.float64)
    ducked_ones = kick_triggered_sidechain(
        ones, kick_samples, SR, depth_db=6.0, attack_ms=5.0,
        release_ms=120.0, lookahead_ms=2.0,
    )
    env_min = float(np.min(ducked_ones))
    env_end = float(ducked_ones[0, -1])
    print(f"[3] kick_triggered_sidechain: envelope min {env_min:.4f} "
          f"(expect {10 ** (-6 / 20):.4f}) | envelope at end {env_end:.4f} "
          f"(expect 1.0000) | out shape {ducked_ones.shape}")
    assert abs(env_min - 10 ** (-6 / 20)) < 1e-6
    assert abs(env_end - 1.0) < 1e-9
    ducked_tone = kick_triggered_sidechain(tone, kick_samples, SR)
    print(f"    tone peak before {float(np.max(np.abs(tone))):.4f} -> "
          f"after {float(np.max(np.abs(ducked_tone))):.4f}")

    # --- Dynamic vocal carving: vocal = 1 kHz sine active 0.5-1.5 s. ---
    vocal = np.zeros(N, dtype=np.float64)
    vocal[22050:66150] = 0.4 * np.sin(2 * np.pi * 1000.0 * t[22050:66150])
    carved = dynamic_vocal_carve(tone, vocal, SR, bands=8, reduction_db=3.0)
    seg = slice(22050, 66150)
    rms_before = float(np.sqrt(np.mean(tone[:, seg] ** 2)))
    rms_after = float(np.sqrt(np.mean(carved[:, seg] ** 2)))
    print(f"[4] dynamic_vocal_carve: out shape {carved.shape}, dtype {carved.dtype} "
          f"| RMS in vocal region {rms_before:.4f} -> {rms_after:.4f} "
          f"(carved down: {rms_after < rms_before})")
    assert carved.shape == tone.shape and carved.dtype == tone.dtype
    assert rms_after < rms_before

    # --- Formant-preserving time-stretch. ---
    stretched = time_stretch_formant_preserving(tone, SR, rate=1.25)
    expect_len = int(round(N / 1.25))
    print(f"[5] time_stretch_formant_preserving rate=1.25: in {N} -> "
          f"out {stretched.shape[1]} samples (expect {expect_len})")
    assert stretched.shape == (2, expect_len)
    try:
        time_stretch_formant_preserving(tone, SR, rate=0.0)
    except ValueError as exc:
        print(f"    rate validation OK (ValueError: {exc})")
    else:  # pragma: no cover
        raise AssertionError("rate=0 did not raise ValueError")

    # --- Binaural pan. ---
    panned = binaural_pan(tone_mono, SR, azimuth_deg=30.0)
    rms_l = float(np.sqrt(np.mean(panned[0] ** 2)))
    rms_r = float(np.sqrt(np.mean(panned[1] ** 2)))
    panned_hard_left = binaural_pan(tone_mono, SR, azimuth_deg=-90.0)
    print(f"[6] binaural_pan +30 deg: out {panned.shape} | "
          f"RMS L {rms_l:.4f} / R {rms_r:.4f} (right louder: {rms_r > rms_l}) | "
          f"hard-left right RMS {float(np.sqrt(np.mean(panned_hard_left[1] ** 2))):.6f}")
    assert panned.shape == (2, N) and rms_r > rms_l
    assert float(np.sqrt(np.mean(panned_hard_left[1] ** 2))) < 1e-9

    # --- Loudness + clipping. ---
    m = measure_loudness(tone, SR)
    print(f"[7] measure_loudness: integrated {m['integrated_lufs']:.2f} LUFS | "
          f"true peak {m['true_peak_dbtp']:.2f} dBTP | peak {m['peak_dbfs']:.2f} dBFS | "
          f"RMS {m['rms_db']:.2f} dB | clipping {m['clipping']}")
    clean = detect_clipping(tone)
    hot = tone.copy()
    hot[:, 1000:1010] = 1.0
    clipped = detect_clipping(hot)
    print(f"[8] detect_clipping: clean tone {clean['clipped_samples']} clipped "
          f"({clean['percentage']:.3f}%) | hot tone {clipped['clipped_samples']} clipped "
          f"({clipped['percentage']:.3f}%), first positions {clipped['positions'][:5]}")
    assert clean["clipped_samples"] == 0
    assert clipped["clipped_samples"] == 10

    # --- LUFS normalization. ---
    normed = normalize_to_lufs(tone, SR, target_lufs=-14.0)
    m_norm = measure_loudness(normed, SR)
    print(f"[9] normalize_to_lufs(-14): measured {m_norm['integrated_lufs']:.2f} LUFS "
          f"(target -14.00) | true peak {m_norm['true_peak_dbtp']:.2f} dBTP "
          f"(ceiling -1.0)")
    assert abs(m_norm["integrated_lufs"] - (-14.0)) < 0.5
    assert m_norm["true_peak_dbtp"] <= -1.0
    # Ceiling path: sparse loud bursts (high crest factor) pushed to -9 LUFS
    # would peak far above -1 dBTP without the ceiling, so the ceiling
    # must win and the final loudness must sit below the target.
    peaky = np.zeros((2, N), dtype=np.float64)
    burst = 0.99 * np.sin(2 * np.pi * 440.0 * t[:441])
    for start in (0, 22050, 44100, 66150):
        peaky[:, start:start + 441] = burst
    normed_hot = normalize_to_lufs(peaky, SR, target_lufs=-9.0)
    m_hot = measure_loudness(normed_hot, SR)
    print(f"    ceiling path: measured {m_hot['integrated_lufs']:.2f} LUFS "
          f"(below target -9.00: {m_hot['integrated_lufs'] < -9.0}) | "
          f"true peak {m_hot['true_peak_dbtp']:.2f} dBTP (<= -1.0: "
          f"{m_hot['true_peak_dbtp'] <= -1.0})")
    assert m_hot["true_peak_dbtp"] <= -1.0
    assert m_hot["integrated_lufs"] < -9.0

    print("== self-test PASSED ==")
