"""A&R Creative Director & Viral Hook agent for the Autonomous Remix Studio.

This is the LangGraph node that turns a raw song into the strict
:class:`~agents.state.ArrangementBlueprint` every later agent works from:
where the Intro / Build-up / Drop / Breakdown / Outro fall (exact
millisecond timestamps), which 15 s / 30 s windows will retain viewers on
Reels/Shorts, how genres fuse, and which moments fire a drop.

How it works (no LLM required — the heuristic path is complete on its own):

* **Tempo** — ``librosa.feature.tempo`` on the onset envelope (with
  ``librosa.beat.beat_track`` as second attempt), falling back to
  the state's BPM or 120 BPM only when both estimators genuinely fail
  — and the fallback is recorded honestly in ``notes`` (D11: the old
  ``librosa.beat.tempo`` call does not exist in installed librosa and
  silently returned 120.0 for every song). Section boundaries are then
  snapped to the bar grid so every section's ``bars`` field agrees
  with its duration.
* **Energy / novelty** — a ~100 ms RMS envelope (50 ms hop) plus spectral
  novelty (onset strength) on the same grid. The Drop is the longest
  sustained high-energy run, refined to the nearest onset transient; the
  Intro ends at the first sustained rise, the Breakdown is the low-energy
  region after the Drop, the Outro is the tail.
* **Viral hooks** — sliding-window mean energy; the best 15 s and 30 s
  windows become the retention slices.
* **Genre fusion** — keywords parsed out of ``creative_direction``. Tamil /
  South Indian / folk direction defaults to the studio signature
  Kuthu + Phonk blend.
* **Drop triggers** — WhisperX word-level lyric timestamps are not
  installed yet, so triggers are built from the strongest energy peaks
  instead (``trigger_type="energy_peak"`` semantics, ``word`` set to a
  ``"<section>-peak-<ms>ms"`` label, confidence from peak prominence).
  No lyric words are invented. When ``mcp_servers/stem_theory_server``
  lands, replace :func:`_energy_triggers` with WhisperX alignment and
  the rest of this module is unchanged.
* **LLM hook** — :func:`build_blueprint` accepts an optional ``llm``
  callable (``llm(prompt) -> str``). When given, it is asked once for
  one-line production notes per section, parsed defensively as JSON; any
  failure is ignored and the heuristic descriptions stand.

Honest limits, recorded in the run's ``notes`` channel:

* The key is estimated from a chroma / Krumhansl-Schmuckler match when no
  key is supplied — it is a guess, not a transcription.
* A track shorter than 30 s cannot form a *valid* blueprint at all: the
  schema requires a 30 s viral-hook window inside the track bounds. The
  section segmenter still produces all five roles with exact tiling for
  short audio (see :func:`_build_sections`), but :func:`build_blueprint`
  raises a clear ``ValueError`` for tracks under 30 s instead of shipping
  an invalid blueprint.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.state import (  # noqa: E402
    SAMPLE_RATE,
    ArrangementBlueprint,
    GenreFusionRule,
    LyricDropTrigger,
    RemixAgentState,
    Section,
    SectionType,
    ViralHookWindow,
)

__all__ = ["build_blueprint", "director_node"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: RMS analysis window / hop, milliseconds.
_FRAME_MS: int = 100
_HOP_MS: int = 50

#: A valid blueprint needs a 30 s hook window inside the track.
_MIN_TRACK_MS: int = 30_000

#: Section roles in playback order.
_ROLES: tuple[SectionType, ...] = (
    SectionType.INTRO,
    SectionType.BUILD_UP,
    SectionType.DROP,
    SectionType.BREAKDOWN,
    SectionType.OUTRO,
)

#: Keyword (substring, lowercase) -> genre tag, checked in order.
_GENRE_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("south indian", "kuthu"),
    ("south-indian", "kuthu"),
    ("tamil", "kuthu"),
    ("kuthu", "kuthu"),
    ("carnatic", "carnatic"),
    ("hindustani", "hindustani"),
    ("folk", "folk"),
    ("phonk", "phonk"),
    ("trap", "trap"),
    ("cyberpunk", "cyberpunk"),
    ("drum and bass", "drum-and-bass"),
    ("drum-and-bass", "drum-and-bass"),
    ("dnb", "drum-and-bass"),
    ("dubstep", "dubstep"),
    ("house", "house"),
    ("techno", "techno"),
    ("trance", "trance"),
    ("hip hop", "hip-hop"),
    ("hip-hop", "hip-hop"),
    ("lo-fi", "lofi"),
    ("lofi", "lofi"),
    ("synthwave", "synthwave"),
    ("afrobeat", "afrobeat"),
    ("reggaeton", "reggaeton"),
    ("baile funk", "baile-funk"),
    ("moombahton", "moombahton"),
    ("ambient", "ambient"),
    ("edm", "edm"),
)

_RAGA_KEYWORDS: dict[str, str] = {
    "mohanam": "Mohanam",
    "kalyani": "Kalyani",
    "shankarabharanam": "Shankarabharanam",
    "hindolam": "Hindolam",
    "harikambhoji": "Harikambhoji",
}

_TALA_KEYWORDS: dict[str, str] = {
    "adi": "Adi (8-beat)",
    "rupakam": "Rupakam (3-beat)",
    "misra chapu": "Misra Chapu (7-beat)",
    "khanda chapu": "Khanda Chapu (5-beat)",
}

#: Krumhansl-Schmuckler key profiles (major / minor).
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_ROLE_DESCRIPTIONS: dict[SectionType, str] = {
    SectionType.INTRO: "Sparse opening — establish the motif before the build.",
    SectionType.BUILD_UP: "Rising tension — groove layers in, textures enter.",
    SectionType.DROP: "Full impact — the peak-energy moment of the track.",
    SectionType.BREAKDOWN: "Pull back to the motif — space for the next lift.",
    SectionType.OUTRO: "Tail out on the motif — clean ending for the next track.",
}


# ---------------------------------------------------------------------------
# Audio analysis helpers
# ---------------------------------------------------------------------------


def _load_mono(path: str) -> tuple[np.ndarray, int]:
    """Load audio as a mono float64 array; the studio never resamples."""
    info = sf.info(path)
    if info.samplerate != SAMPLE_RATE:
        raise ValueError(
            f"expected {SAMPLE_RATE} Hz audio, got {info.samplerate} Hz — "
            "the studio never resamples mid-pipeline"
        )
    audio, _sr = sf.read(path, always_2d=False)
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, info.samplerate


def _estimate_tempo(
    y: np.ndarray, sr: int, fallback: float = 120.0
) -> tuple[float, bool]:
    """Tempo from the onset envelope.

    Returns ``(tempo, used_fallback)``. Primary path is
    ``librosa.feature.tempo`` (the successor of the removed
    ``librosa.beat.tempo`` — D11: that name does not exist in the
    installed librosa 1.0.1, so the old call raised ``AttributeError``
    and the ``except`` silently returned 120.0 for EVERY song). The
    secondary path is ``librosa.beat.beat_track``. The numeric
    ``fallback`` is used ONLY when both real estimators fail, and the
    caller must record that honestly in ``notes`` — a silent 120.0
    makes the whole pipeline self-consistent on a wrong tempo.
    """
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.feature.tempo(
            onset_envelope=onset_env, sr=sr, aggregate=np.median
        )
        value = float(np.atleast_1d(tempo)[0])
        if math.isfinite(value) and value > 0:
            return float(min(max(value, 40.0), 240.0)), False
    except Exception:
        pass
    try:
        tempo, _beats = librosa.beat.beat_track(y=y, sr=sr)
        value = float(np.atleast_1d(tempo)[0])
        if math.isfinite(value) and value > 0:
            return float(min(max(value, 40.0), 240.0)), False
    except Exception:
        pass
    return fallback, True


def _estimate_key(y: np.ndarray, sr: int) -> tuple[str, str]:
    """Rough key from chroma vs Krumhansl-Schmuckler profiles.

    Returns ``(key, mode)`` e.g. ``("A minor", "minor")``. This is an
    estimate, never a transcription — callers must say so in ``notes``.
    """
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        vec = chroma.mean(axis=1)
        if not np.any(vec):
            return "A minor", "minor"
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        best_key, best_mode, best_score = "A minor", "minor", -1.0
        for root in range(12):
            for mode, profile in (
                ("major", _MAJOR_PROFILE),
                ("minor", _MINOR_PROFILE),
            ):
                rotated = np.roll(profile, root)
                rotated = rotated / (np.linalg.norm(rotated) + 1e-9)
                score = float(np.dot(vec, rotated))
                if score > best_score:
                    best_key, best_mode, best_score = (
                        f"{_PITCH_CLASSES[root]} {mode}",
                        mode,
                        score,
                    )
        return best_key, best_mode
    except Exception:
        return "A minor", "minor"


def _envelopes(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RMS envelope (smoothed), onset strength, and frame times in ms.

    RMS uses ~100 ms windows on a 50 ms hop; the moving average is
    edge-normalised so the Intro/Outro are not artificially attenuated.
    Onset strength is computed on the same hop so the grids line up.
    """
    frame = int(sr * _FRAME_MS / 1000)
    hop = int(sr * _HOP_MS / 1000)
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    times_ms = librosa.frames_to_time(
        np.arange(len(rms)), sr=sr, hop_length=hop
    ) * 1000.0
    win = max(1, int(500 / _HOP_MS))
    kernel = np.ones(win) / win
    smooth = np.convolve(rms, kernel, mode="same") / np.convolve(
        np.ones_like(rms), kernel, mode="same"
    )
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    if len(onset) > len(rms):
        onset = onset[: len(rms)]
    elif len(onset) < len(rms):
        onset = np.pad(onset, (0, len(rms) - len(onset)), mode="edge")
    return times_ms, smooth, onset


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def _longest_run(mask: np.ndarray) -> tuple[int, int] | None:
    """Longest contiguous run of True in ``mask`` as (start, end) indices."""
    best: tuple[int, int] | None = None
    start: int | None = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if best is None or (i - 1) - start > best[1] - best[0]:
                best = (start, i - 1)
            start = None
    if start is not None and (
        best is None or (len(mask) - 1) - start > best[1] - best[0]
    ):
        best = (start, len(mask) - 1)
    return best


def _nearest_onset(onset: np.ndarray, index: int, window: int) -> int:
    """Index of the strongest onset within ±``window`` frames of ``index``."""
    lo = max(0, index - window)
    hi = min(len(onset), index + window + 1)
    if hi <= lo:
        return index
    return lo + int(np.argmax(onset[lo:hi]))


def _proportional_boundaries(duration_ms: int) -> list[int]:
    """Fallback split: Intro 15%, Build-up 20%, Drop 30%, Breakdown 20%, Outro 15%."""
    fracs = [0.0, 0.15, 0.35, 0.65, 0.85, 1.0]
    return [int(round(f * duration_ms)) for f in fracs]


def _segment_boundaries(
    times_ms: np.ndarray,
    smooth: np.ndarray,
    onset: np.ndarray,
    duration_ms: int,
    bar_ms: float,
) -> list[int]:
    """Six boundaries (ms) tiling the track into the five roles.

    Energy-derived when the profile is informative, proportional otherwise.
    Internal boundaries are snapped to the bar grid; the first is 0 and the
    last is exactly ``duration_ms``. Every section spans at least one bar
    (D8) — a 250 ms build-up is schema-valid but musically absurd — except
    on tracks too short to hold five full bars, where the gap is the
    largest value that still lets five sections fit.
    """
    n = len(smooth)
    hop_ms = float(times_ms[1] - times_ms[0]) if n > 1 else float(_HOP_MS)
    gap_frames = max(1, int(250 / hop_ms))
    raw: list[int] | None = None

    if n >= 20:
        peak = float(np.percentile(smooth, 99))
        baseline = float(np.percentile(smooth, 20))
        span = max(peak - baseline, 1e-6)
        hi = baseline + 0.70 * span
        lo = baseline + 0.35 * span

        run = _longest_run(smooth >= hi)
        if run is None:
            drop_s, drop_e = int(n * 0.375), int(n * 0.625)
        else:
            drop_s, drop_e = run
            max_len = max(1, int(0.40 * n))
            if drop_e - drop_s + 1 > max_len:
                peak_i = int(np.argmax(smooth))
                drop_s = max(0, peak_i - max_len // 2)
                drop_e = min(n - 1, drop_s + max_len - 1)
            min_len = max(1, int(1000 / hop_ms))
            if drop_e - drop_s + 1 < min_len:
                peak_i = int(np.argmax(smooth))
                half = min_len // 2
                drop_s = max(0, peak_i - half)
                drop_e = min(n - 1, peak_i + half)

        # Refine the drop start to the nearest onset transient (±0.5 s).
        drop_s = _nearest_onset(onset, drop_s, window=int(500 / hop_ms))
        drop_s = max(0, min(drop_s, n - 1))

        degenerate = drop_s < 3 * gap_frames or drop_e > n - 3 * gap_frames
        if not degenerate:
            intro_e: int | None = None
            for i in range(0, drop_s):
                if smooth[i] >= lo:
                    intro_e = i
                    break
            if intro_e is None or intro_e == 0:
                intro_e = max(int(0.10 * n), 1)
            intro_e = min(intro_e, drop_s - gap_frames)

            limit = int(0.90 * n)
            breakdown_e: int | None = None
            for i in range(min(limit, n - 1), drop_e, -1):
                if smooth[i] <= lo:
                    breakdown_e = i
                    break
            if breakdown_e is None:
                breakdown_e = min(drop_e + int(0.20 * n), limit)
            breakdown_e = max(breakdown_e, drop_e + gap_frames)
            breakdown_e = min(breakdown_e, n - gap_frames)

            if 0 < intro_e < drop_s < drop_e < breakdown_e < n:
                raw = [
                    0,
                    int(round(times_ms[intro_e])),
                    int(round(times_ms[drop_s])),
                    int(round(times_ms[drop_e])),
                    int(round(times_ms[breakdown_e])),
                    duration_ms,
                ]

    if raw is None:
        raw = _proportional_boundaries(duration_ms)

    # Snap internal boundaries to the bar grid, keep them strictly
    # increasing with at least one bar per section (D8), exact end
    # preserved. The gap is capped at duration_ms // 5 so five sections
    # always fit, even on a track shorter than five bars.
    min_gap_ms = max(1, min(int(round(bar_ms)), duration_ms // 5))
    bounds = [0]
    for i in range(1, 5):
        lo_ms = bounds[-1] + min_gap_ms
        hi_ms = duration_ms - (5 - i) * min_gap_ms
        snapped = int(round(round(raw[i] / bar_ms) * bar_ms))
        bounds.append(max(lo_ms, min(hi_ms, snapped)))
    bounds.append(duration_ms)
    return bounds


def _build_sections(
    times_ms: np.ndarray,
    smooth: np.ndarray,
    onset: np.ndarray,
    duration_ms: int,
    bar_ms: float,
    bpm: float,
    key: str,
    genre_pair: list[str],
) -> list[Section]:
    """Build the five tiling sections for any track length.

    Works for short audio too. Every section spans at least one bar
    (enforced by :func:`_segment_boundaries`) whenever the track holds
    five full bars; on shorter tracks the gap is the largest value that
    still lets five sections fit, and the schema's one-bar tolerance
    absorbs the remainder.
    """
    boundaries = _segment_boundaries(times_ms, smooth, onset, duration_ms, bar_ms)
    peak = float(np.percentile(smooth, 99)) if len(smooth) else 1.0
    sections: list[Section] = []
    for role, start_ms, end_ms in zip(_ROLES, boundaries[:-1], boundaries[1:]):
        mask = (times_ms >= start_ms) & (times_ms < end_ms)
        energy = float(np.mean(smooth[mask]) / peak) if np.any(mask) else 0.1
        energy = float(min(max(energy, 0.05), 1.0))
        bars = max(1, int(round((end_ms - start_ms) / bar_ms)))
        sections.append(
            Section(
                name=role,
                start_ms=int(start_ms),
                end_ms=int(end_ms),
                energy=round(energy, 3),
                bpm=round(bpm, 2),
                key=key,
                bars=bars,
                genre_fusion=list(genre_pair),
                description=_ROLE_DESCRIPTIONS[role],
            )
        )
    return sections


# ---------------------------------------------------------------------------
# Hooks, fusion, triggers
# ---------------------------------------------------------------------------


def _best_window(
    times_ms: np.ndarray,
    smooth: np.ndarray,
    duration_ms: int,
    window_s: int,
    peak_ref: float,
) -> tuple[int, float]:
    """Start (ms) and retention score of the highest-mean-energy window."""
    win_frames = int(window_s * 1000 / _HOP_MS)
    if len(smooth) < win_frames:
        # Track barely longer than the window: the whole track is the hook.
        score = float(min(max(float(np.mean(smooth)) / peak_ref, 0.0), 1.0))
        return 0, score
    means = np.convolve(smooth, np.ones(win_frames) / win_frames, mode="valid")
    j = int(np.argmax(means))
    start_ms = int(round(times_ms[j]))
    start_ms = min(start_ms, duration_ms - window_s * 1000)
    score = float(min(max(means[j] / peak_ref, 0.0), 1.0))
    return start_ms, round(score, 3)


def _resolve_genres(direction: str) -> list[str]:
    """Two genre tags from the direction; Kuthu + Phonk is the default."""
    text = direction.lower()
    found: list[str] = []
    for keyword, tag in _GENRE_KEYWORDS:
        if keyword in text and tag not in found:
            found.append(tag)
    if not found:
        return ["kuthu", "phonk"]
    if len(found) == 1:
        partner = "trap" if found[0] == "phonk" else "phonk"
        found.append(partner)
    return found[:2]


def _resolve_raga_tala(direction: str) -> tuple[str | None, str | None]:
    """Raga / tala only when the direction names one — never guessed."""
    text = direction.lower()
    raga = next(
        (name for keyword, name in _RAGA_KEYWORDS.items() if keyword in text), None
    )
    tala = next(
        (name for keyword, name in _TALA_KEYWORDS.items() if keyword in text), None
    )
    return raga, tala


def _energy_triggers(
    times_ms: np.ndarray,
    smooth: np.ndarray,
    sections: list[Section],
    duration_ms: int,
) -> list[LyricDropTrigger]:
    """Drop triggers from the strongest energy peaks.

    Energy-peak triggers, in the exact shape WhisperX word-level triggers
    will take: same model, same validation, but the ``word`` is an honest
    label and confidence comes from peak prominence — no lyric words are
    invented. Swap this function for WhisperX alignment when
    ``stem_theory_server`` lands.
    """
    if len(smooth) == 0:
        return []
    peak = float(np.percentile(smooth, 99))
    baseline = float(np.percentile(smooth, 20))
    span = max(peak - baseline, 1e-6)
    distance = max(1, int(2000 / _HOP_MS))
    try:
        peaks, props = find_peaks(
            smooth, distance=distance, prominence=0.10 * span
        )
        prominences = np.asarray(props.get("prominences", []), dtype=float)
    except Exception:
        peaks, prominences = np.array([], dtype=int), np.array([], dtype=float)
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(smooth))])
        prominences = np.array([span])

    def section_at(ms: int) -> Section:
        for section in sections:
            if section.start_ms <= ms < section.end_ms:
                return section
        return sections[-1]

    order = np.argsort(smooth[peaks])[::-1][:3]
    triggers: list[LyricDropTrigger] = []
    for rank in order:
        peak_idx = int(peaks[rank])
        peak_ms = min(int(round(times_ms[peak_idx])), duration_ms - 1)
        section = section_at(peak_ms)
        start_ms = peak_ms
        end_ms = min(peak_ms + 250, section.end_ms)
        if end_ms <= start_ms:
            end_ms = section.end_ms
            start_ms = max(section.start_ms, end_ms - 250)
            if end_ms <= start_ms:
                continue
        confidence = float(
            min(max(float(prominences[rank]) / span, 0.05), 0.99)
        )
        trigger_type = (
            "drop"
            if section.name == SectionType.DROP
            else "riser"
            if section.name == SectionType.BUILD_UP
            else "impact"
        )
        triggers.append(
            LyricDropTrigger(
                word=f"{section.name.value}-peak-{peak_ms}ms",
                word_start_ms=start_ms,
                word_end_ms=end_ms,
                trigger=trigger_type,  # type: ignore[arg-type]
                confidence=round(confidence, 3),
                section=section.name,
            )
        )
    triggers.sort(key=lambda t: t.word_start_ms)
    return triggers


def _refine_with_llm(
    llm: Callable[..., Any] | None, blueprint: ArrangementBlueprint
) -> None:
    """Optionally let an LLM rewrite section descriptions.

    The callable is invoked once as ``llm(prompt) -> str`` and the reply
    must be JSON mapping section role -> one-line note. Anything else —
    wrong type, bad JSON, exception — is silently ignored and the
    heuristic descriptions stand.
    """
    if llm is None:
        return
    try:
        summary = {
            "title": blueprint.title,
            "bpm": blueprint.bpm,
            "key": blueprint.key,
            "sections": [
                {
                    "role": s.name.value,
                    "start_ms": s.start_ms,
                    "end_ms": s.end_ms,
                    "energy": s.energy,
                }
                for s in blueprint.sections
            ],
            "hooks": [
                {"duration_s": h.duration_s, "start_ms": h.start_ms}
                for h in blueprint.viral_hooks
            ],
        }
        prompt = (
            "You are an A&R director. Given this remix arrangement JSON, "
            "return ONLY a JSON object mapping each section role to a "
            "one-line production note (max 120 chars). No other text.\n"
            + json.dumps(summary)
        )
        raw = llm(prompt)
        if not isinstance(raw, str):
            return
        data = json.loads(raw)
        if not isinstance(data, dict):
            return
        for section in blueprint.sections:
            note = data.get(section.name.value)
            if isinstance(note, str) and note.strip():
                section.description = note.strip()[:200]
    except Exception:
        return


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_blueprint(
    audio_path: str,
    creative_direction: str = "",
    bpm: float | None = None,
    key: str | None = None,
    llm: Callable[..., Any] | None = None,
    notes: list[str] | None = None,
) -> ArrangementBlueprint:
    """Analyse ``audio_path`` and return a validated ArrangementBlueprint.

    ``bpm`` / ``key`` override estimation when given. ``notes`` (optional)
    is filled with diagnostic strings — estimation caveats, the WhisperX
    status — for the caller to forward to the state's ``notes`` channel.

    Raises ``FileNotFoundError`` for a missing file, ``ValueError`` for
    non-44100 Hz audio or tracks shorter than 30 s (the schema requires a
    30 s viral-hook window inside the track).
    """
    diagnostics: list[str] = [] if notes is None else notes
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"input audio not found: {audio_path!r}")

    y, sr = _load_mono(str(path))
    duration_ms = int(round(len(y) / sr * 1000))
    if duration_ms < _MIN_TRACK_MS:
        raise ValueError(
            f"track is {duration_ms / 1000:.1f} s; ArrangementBlueprint "
            "requires >= 30 s for the 30 s viral-hook window"
        )

    if bpm is None:
        tempo, used_fallback = _estimate_tempo(y, sr)
        if used_fallback:
            diagnostics.append(
                "director: tempo estimation failed — fell back to "
                f"{tempo:.1f} BPM (no BPM supplied); blueprint grid may "
                "not match the song's true tempo"
            )
        else:
            diagnostics.append(
                f"director: tempo estimated with librosa at {tempo:.1f} BPM "
                "(no BPM supplied)"
            )
    else:
        tempo = float(min(max(bpm, 40.0), 240.0))

    if key is None:
        key_name, mode = _estimate_key(y, sr)
        diagnostics.append(
            f"director: key estimated from chroma as {key_name!r} "
            "(no key supplied — estimate, not transcription)"
        )
    else:
        key_name = key
        mode = "minor" if "minor" in key_name.lower() else "major"

    times_ms, smooth, onset = _envelopes(y, sr)
    bar_ms = 240000.0 / tempo
    genre_pair = _resolve_genres(creative_direction)
    raga, tala = _resolve_raga_tala(creative_direction)

    sections = _build_sections(
        times_ms, smooth, onset, duration_ms, bar_ms, tempo, key_name, genre_pair
    )

    peak_ref = float(np.percentile(smooth, 99)) if len(smooth) else 1.0
    hooks: list[ViralHookWindow] = []
    for window_s in (30, 15):
        start_ms, score = _best_window(
            times_ms, smooth, duration_ms, window_s, peak_ref
        )
        hooks.append(
            ViralHookWindow(
                duration_s=window_s,  # type: ignore[arg-type]
                start_ms=start_ms,
                end_ms=start_ms + window_s * 1000,
                retention_score=score,
                reason=(
                    f"Highest-mean-energy {window_s} s window "
                    f"({score:.2f} normalised) — covers the main drop."
                ),
            )
        )
    hooks.sort(key=lambda h: h.duration_s)

    triggers = _energy_triggers(times_ms, smooth, sections, duration_ms)
    if triggers:
        diagnostics.append(
            "director: WhisperX not installed — drop triggers are energy "
            "peaks, not lyric words (see _energy_triggers docstring)"
        )

    blueprint = ArrangementBlueprint(
        title=f"{path.stem} — Remix",
        track_duration_ms=duration_ms,
        bpm=round(tempo, 2),
        key=key_name,
        sections=sections,
        viral_hooks=hooks,
        sample_rate=SAMPLE_RATE,
        scale=mode,
        raga=raga,
        tala=tala,
        drop_triggers=triggers,
        genre_fusion_rules=[
            GenreFusionRule(
                genres=list(genre_pair),
                blend_ratio=0.6,
                applies_to=[SectionType.BUILD_UP, SectionType.DROP],
                notes=(
                    f"{genre_pair[0]} leads the groove; {genre_pair[1]} "
                    "supplies the texture and low-end weight."
                ),
            )
        ],
        creative_direction=creative_direction,
    )
    _refine_with_llm(llm, blueprint)
    return blueprint


def director_node(
    state: RemixAgentState,
    llm: Callable[..., Any] | None = None,
) -> dict:
    """LangGraph node: fill ``blueprint`` (plus bpm/key when absent).

    Returns a partial state update ``{"blueprint": ...}``. On failure the
    update carries only ``errors`` and no blueprint — the graph can route
    to a failure branch instead of crashing.
    """
    audio_path = state.get("input_audio_path") or ""
    if not audio_path or not os.path.isfile(audio_path):
        return {"errors": [f"director: input audio not found: {audio_path!r}"]}

    diagnostics: list[str] = []
    try:
        blueprint = build_blueprint(
            audio_path,
            creative_direction=state.get("creative_direction") or "",
            bpm=state.get("bpm"),
            key=state.get("key"),
            llm=llm,
            notes=diagnostics,
        )
    except Exception as exc:  # noqa: BLE001 — surfaced to the graph as data
        return {
            "errors": [
                f"director: failed to build blueprint for {audio_path!r}: {exc}"
            ]
        }

    update: dict[str, Any] = {"blueprint": blueprint}
    if state.get("bpm") is None:
        update["bpm"] = blueprint.bpm
    if state.get("key") is None:
        update["key"] = blueprint.key
    if diagnostics:
        update["notes"] = diagnostics
    return update


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Director agent: build a blueprint.")
    parser.add_argument("audio", help="Input WAV path (44100 Hz).")
    parser.add_argument("--direction", default="", help="Creative direction text.")
    parser.add_argument("--bpm", type=float, default=None, help="Override tempo.")
    parser.add_argument("--key", default=None, help="Override key, e.g. 'A minor'.")
    args = parser.parse_args()
    print(build_blueprint(args.audio, args.direction, args.bpm, args.key).model_dump_json(indent=2))
