"""State and schema layer for the Autonomous Multi-Agent AI Music Producer
& Remix Studio.

This module is the single source of truth for data that flows through the
LangGraph workflow:

* :class:`ArrangementBlueprint` — the A&R Director agent's strict JSON
  contract. Every timestamp is an integer number of **milliseconds** and
  sections must tile the track exactly (no gaps, no overlaps).
* :class:`RemixAgentState` — the LangGraph ``TypedDict`` carried from node
  to node. The ``errors`` channel uses an ``operator.add`` reducer so
  parallel agents accumulate diagnostics instead of overwriting them.
* :class:`CriticReport` — the self-correction loop's structured verdict.

Global invariants enforced here:

* Sample rate is **44 100 Hz everywhere** (``SAMPLE_RATE``). A blueprint
  or state built at any other rate is rejected at construction time.
* The critic loop runs at most ``MAX_CRITIC_ITERATIONS`` (3) re-renders.
* A blueprint is only valid when its sections start at 0 ms, end exactly
  at the track duration, and contain at least one Drop section.
"""

from __future__ import annotations

import operator
import re
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator, model_validator

__all__ = [
    "SAMPLE_RATE",
    "MAX_CRITIC_ITERATIONS",
    "BEATS_PER_BAR",
    "KNOWN_GENRE_TAGS",
    "SectionType",
    "Section",
    "GenreFusionRule",
    "LyricDropTrigger",
    "ViralHookWindow",
    "ArrangementBlueprint",
    "CriticReport",
    "RemixAgentState",
    "create_initial_state",
]

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

#: The one sample rate used by every agent, tool and export in the studio.
SAMPLE_RATE: int = 44100

#: Hard cap on critic -> re-render iterations before the run is forced to
#: stop and report failure instead of looping forever.
MAX_CRITIC_ITERATIONS: int = 3

#: Arrangements are built on a 4/4 beat grid unless a section says otherwise.
BEATS_PER_BAR: int = 4

#: Reference vocabulary for ``genre_fusion`` tags. Tags are validated for
#: shape (lowercase slug) rather than membership, so the Director agent may
#: coin new fusions, but it should prefer these.
KNOWN_GENRE_TAGS: frozenset[str] = frozenset(
    {
        "kuthu",
        "folk",
        "carnatic",
        "hindustani",
        "phonk",
        "trap",
        "cyberpunk",
        "edm",
        "house",
        "techno",
        "trance",
        "dubstep",
        "drum-and-bass",
        "hip-hop",
        "lofi",
        "afrobeat",
        "reggaeton",
        "baile-funk",
        "moombahton",
        "ambient",
        "synthwave",
    }
)

_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9\-]*$")

_KEY_RE = re.compile(
    r"^([A-Ga-g])([#b]?)\s+"
    r"(major|minor|dorian|phrygian|lydian|mixolydian|locrian|ionian|aeolian)$"
)


def _normalize_genre_tag(value: str) -> str:
    """Normalize one genre tag to a lowercase slug (``"Drum & Bass"`` -> ``"drum-bass"``)."""
    tag = value.strip().lower().replace(" ", "-").replace("_", "-").replace("&", "")
    tag = re.sub(r"-+", "-", tag).strip("-")
    if not tag or not _TAG_RE.match(tag):
        raise ValueError(f"invalid genre tag: {value!r} (expected a lowercase slug)")
    return tag


def _normalize_key(value: str) -> str:
    """Validate and normalize a musical key, e.g. ``"a minor"`` -> ``"A minor"``."""
    match = _KEY_RE.match(value.strip())
    if match is None:
        raise ValueError(
            f"invalid key: {value!r} "
            "(expected e.g. 'A minor', 'C major', 'F# dorian', 'Bb mixolydian')"
        )
    root, accidental, mode = match.groups()
    return f"{root.upper()}{accidental} {mode.lower()}"


# ---------------------------------------------------------------------------
# Arrangement building blocks
# ---------------------------------------------------------------------------


class SectionType(str, Enum):
    """The five structural roles a remix section can play, in story order."""

    INTRO = "intro"
    BUILD_UP = "build_up"
    DROP = "drop"
    BREAKDOWN = "breakdown"
    OUTRO = "outro"

    @property
    def display_name(self) -> str:
        """Human-readable label, e.g. ``SectionType.BUILD_UP.display_name == "Build-up"``."""
        return self.value.replace("_", "-").title().replace("Build-Up", "Build-up")


class Section(BaseModel):
    """One contiguous, bar-aligned slice of the arrangement.

    Timestamps are integer milliseconds. ``bars`` must agree with the
    section duration at the section BPM (within one bar of tolerance, to
    absorb millisecond rounding of the beat grid).
    """

    model_config = {"validate_assignment": True}

    name: SectionType = Field(description="Structural role of this section.")
    start_ms: int = Field(ge=0, description="Section start, milliseconds from track start.")
    end_ms: int = Field(gt=0, description="Section end (exclusive), milliseconds from track start.")
    energy: float = Field(
        ge=0.0,
        le=1.0,
        description="Perceived energy 0.0 (sparse/quiet) to 1.0 (full drop).",
    )
    bpm: float = Field(gt=20.0, le=300.0, description="Tempo of this section in beats per minute.")
    key: str = Field(description="Musical key, e.g. 'A minor'.")
    bars: int = Field(ge=1, description="Length in 4/4 bars; must match the duration at bpm.")
    genre_fusion: list[str] = Field(
        default_factory=list,
        min_length=1,
        description="Genre tags fused in this section, e.g. ['kuthu', 'phonk'].",
    )
    description: str = Field(
        default="",
        description="Short production note for this section.",
    )

    @field_validator("key", mode="before")
    @classmethod
    def _validate_key(cls, value: str) -> str:
        return _normalize_key(value)

    @field_validator("genre_fusion", mode="before")
    @classmethod
    def _normalize_tags(cls, tags: list[str]) -> list[str]:
        return [_normalize_genre_tag(t) for t in tags]

    @model_validator(mode="after")
    def _check_span_and_bars(self) -> "Section":
        if self.end_ms <= self.start_ms:
            raise ValueError(
                f"section {self.name.value}: end_ms ({self.end_ms}) must exceed "
                f"start_ms ({self.start_ms})"
            )
        beat_ms = 60_000.0 / self.bpm
        expected_ms = self.bars * BEATS_PER_BAR * beat_ms
        actual_ms = self.end_ms - self.start_ms
        tolerance_ms = max(0.10 * expected_ms, BEATS_PER_BAR * beat_ms)
        if abs(actual_ms - expected_ms) > tolerance_ms:
            raise ValueError(
                f"section {self.name.value}: {self.bars} bars at {self.bpm} BPM "
                f"should span ~{expected_ms:.0f} ms, got {actual_ms} ms"
            )
        return self

    @property
    def duration_ms(self) -> int:
        """Length of the section in milliseconds."""
        return self.end_ms - self.start_ms


class GenreFusionRule(BaseModel):
    """How two (or more) genres are blended, and where the blend applies."""

    model_config = {"validate_assignment": True}

    genres: list[str] = Field(
        min_length=2,
        description="Genres being fused, in blend order; index 0 is the dominant genre.",
    )
    blend_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Weight of genres[0] in the blend: 1.0 = pure genres[0], 0.0 = pure genres[-1].",
    )
    applies_to: list[SectionType] = Field(
        default_factory=list,
        description="Sections this rule governs; empty means the whole track.",
    )
    notes: str = Field(default="", description="Production note, e.g. instrumentation guidance.")

    @field_validator("genres", mode="before")
    @classmethod
    def _normalize_tags(cls, tags: list[str]) -> list[str]:
        return [_normalize_genre_tag(t) for t in tags]

    @model_validator(mode="after")
    def _check_distinct_genres(self) -> "GenreFusionRule":
        if len(set(self.genres)) != len(self.genres):
            raise ValueError(f"genre fusion rule lists a genre twice: {self.genres}")
        return self


class LyricDropTrigger(BaseModel):
    """A word-level lyric timestamp that fires a musical event.

    Timestamps come from WhisperX word-level alignment. The drop (or riser,
    chop, ...) must land on the exact sample of the punchline word, never on
    a rounded second.
    """

    model_config = {"validate_assignment": True}

    word: str = Field(min_length=1, description="The lyric word, as transcribed.")
    word_start_ms: int = Field(ge=0, description="Word onset, milliseconds from track start.")
    word_end_ms: int = Field(gt=0, description="Word offset, milliseconds from track start.")
    trigger: Literal["drop", "riser", "chop", "silence", "impact", "filter_sweep"] = Field(
        description="Musical event fired on this word."
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="WhisperX word-alignment confidence."
    )
    section: SectionType | None = Field(
        default=None,
        description="Section the word falls in; validated against the blueprint when set.",
    )

    @model_validator(mode="after")
    def _check_word_span(self) -> "LyricDropTrigger":
        if self.word_end_ms <= self.word_start_ms:
            raise ValueError(
                f"lyric trigger {self.word!r}: word_end_ms ({self.word_end_ms}) "
                f"must exceed word_start_ms ({self.word_start_ms})"
            )
        return self


class ViralHookWindow(BaseModel):
    """A highest-retention slice of the track, cut for Reels/Shorts.

    ``end_ms - start_ms`` must equal ``duration_s * 1000`` exactly, so the
    exported ``viral_hook_30s.wav`` / ``viral_hook_15s.wav`` need no trimming.
    """

    model_config = {"validate_assignment": True}

    duration_s: Literal[15, 30] = Field(description="Hook length in seconds: 15 or 30.")
    start_ms: int = Field(ge=0, description="Hook start, milliseconds from track start.")
    end_ms: int = Field(gt=0, description="Hook end, milliseconds from track start.")
    retention_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Director agent's predicted retention for this window.",
    )
    reason: str = Field(
        default="",
        description="Why this window retains viewers (hook line, drop peak, ...).",
    )

    @model_validator(mode="after")
    def _check_exact_duration(self) -> "ViralHookWindow":
        expected_ms = self.duration_s * 1000
        if self.end_ms - self.start_ms != expected_ms:
            raise ValueError(
                f"viral hook: {self.duration_s}s window must span exactly "
                f"{expected_ms} ms, got {self.end_ms - self.start_ms} ms"
            )
        return self


# ---------------------------------------------------------------------------
# The Arrangement Blueprint (A&R Director agent output)
# ---------------------------------------------------------------------------


class ArrangementBlueprint(BaseModel):
    """Strict JSON contract for a full remix arrangement.

    Sections must tile the track without gaps or overlaps: the first starts
    at 0 ms, each next starts where the previous ends, and the last ends
    exactly at ``track_duration_ms``. At least one 15 s and one 30 s viral
    hook window is required, both inside the track bounds.
    """

    model_config = {"validate_assignment": True}

    title: str = Field(min_length=1, description="Working title of the remix.")
    track_duration_ms: int = Field(gt=0, description="Full track length in milliseconds.")
    bpm: float = Field(gt=20.0, le=300.0, description="Global tempo in beats per minute.")
    key: str = Field(description="Global musical key, e.g. 'A minor'.")
    sections: list[Section] = Field(
        min_length=1, description="Sections tiling the track, in playback order."
    )
    viral_hooks: list[ViralHookWindow] = Field(
        min_length=1, description="Retention windows; must include one 15s and one 30s."
    )
    sample_rate: Literal[44100] = Field(
        default=44100, description="Sample rate in Hz; fixed at 44100 studio-wide."
    )
    scale: str = Field(
        default="",
        description="Scale/mode detail, e.g. 'minor pentatonic', 'Mohanam'.",
    )
    raga: str | None = Field(
        default=None, description="Indian raga of the source melody, when identified."
    )
    tala: str | None = Field(
        default=None, description="Indian tala / rhythmic cycle, e.g. 'Adi (8-beat)'."
    )
    drop_triggers: list[LyricDropTrigger] = Field(
        default_factory=list,
        description="Lyric-word events that fire drops, risers and chops.",
    )
    genre_fusion_rules: list[GenreFusionRule] = Field(
        default_factory=list,
        description="Genre-blend rules governing the arrangement.",
    )
    creative_direction: str = Field(
        default="",
        description="Free-text direction the Director agent was given.",
    )

    @field_validator("key", mode="before")
    @classmethod
    def _validate_key(cls, value: str) -> str:
        return _normalize_key(value)

    @model_validator(mode="after")
    def _validate_arrangement(self) -> "ArrangementBlueprint":
        sections = self.sections
        if sections[0].name != SectionType.INTRO:
            raise ValueError("first section must be Intro")
        if sections[-1].name != SectionType.OUTRO:
            raise ValueError("last section must be Outro")
        if not any(s.name == SectionType.DROP for s in sections):
            raise ValueError("arrangement must contain at least one Drop section")
        if sections[0].start_ms != 0:
            raise ValueError(
                f"first section must start at 0 ms, got {sections[0].start_ms}"
            )
        if sections[-1].end_ms != self.track_duration_ms:
            raise ValueError(
                f"last section must end at track_duration_ms "
                f"({self.track_duration_ms}), got {sections[-1].end_ms}"
            )
        for prev, cur in zip(sections, sections[1:]):
            if prev.name == cur.name:
                raise ValueError(
                    f"consecutive sections share a role ({prev.name.value}); "
                    "merge them into one section"
                )
            if cur.start_ms != prev.end_ms:
                raise ValueError(
                    f"gap/overlap between {prev.name.value} (ends {prev.end_ms} ms) "
                    f"and {cur.name.value} (starts {cur.start_ms} ms)"
                )

        durations = {h.duration_s for h in self.viral_hooks}
        if 15 not in durations or 30 not in durations:
            raise ValueError("viral_hooks must include at least one 15s and one 30s window")
        for hook in self.viral_hooks:
            if hook.end_ms > self.track_duration_ms:
                raise ValueError(
                    f"viral hook ({hook.duration_s}s) ends at {hook.end_ms} ms, "
                    f"beyond track duration {self.track_duration_ms} ms"
                )

        # Spans per section type: a role may repeat (two Drops, two
        # Build-ups), so keep every occurrence, not just the last.
        spans: dict[SectionType, list[tuple[int, int]]] = {}
        for section in sections:
            spans.setdefault(section.name, []).append((section.start_ms, section.end_ms))
        for trigger in self.drop_triggers:
            if trigger.word_end_ms > self.track_duration_ms:
                raise ValueError(
                    f"lyric trigger {trigger.word!r} ends at {trigger.word_end_ms} ms, "
                    f"beyond track duration {self.track_duration_ms} ms"
                )
            if trigger.section is not None:
                inside = any(
                    start <= trigger.word_start_ms and trigger.word_end_ms <= end
                    for start, end in spans[trigger.section]
                )
                if not inside:
                    occurrences = ", ".join(
                        f"{start}-{end} ms" for start, end in spans[trigger.section]
                    )
                    raise ValueError(
                        f"lyric trigger {trigger.word!r} "
                        f"({trigger.word_start_ms}-{trigger.word_end_ms} ms) "
                        f"does not fall inside any {trigger.section.value} section "
                        f"({occurrences})"
                    )

        present = set(spans)
        for rule in self.genre_fusion_rules:
            missing = [s for s in rule.applies_to if s not in present]
            if missing:
                raise ValueError(
                    f"genre fusion rule {rule.genres} applies to absent sections: "
                    f"{[s.value for s in missing]}"
                )
        return self

    @property
    def duration_seconds(self) -> float:
        """Full track length in seconds."""
        return self.track_duration_ms / 1000.0

    @property
    def total_bars(self) -> int:
        """Total bars across all sections."""
        return sum(s.bars for s in self.sections)

    @property
    def drop_sections(self) -> list[Section]:
        """The Drop section(s) of the arrangement, in playback order."""
        return [s for s in self.sections if s.name == SectionType.DROP]

    def section_at(self, ms: int) -> Section:
        """Return the section containing millisecond ``ms`` (``0 <= ms < duration``)."""
        if not 0 <= ms < self.track_duration_ms:
            raise ValueError(
                f"ms={ms} is outside the track [0, {self.track_duration_ms})"
            )
        for section in self.sections:
            if section.start_ms <= ms < section.end_ms:
                return section
        raise AssertionError("unreachable: sections tile the track by validation")


# ---------------------------------------------------------------------------
# Critic report (self-correction loop verdict)
# ---------------------------------------------------------------------------


class CriticReport(BaseModel):
    """Structured verdict from the Multimodal RLHF Audio Critic agent.

    ``verdict == "approved"`` is only legal with an empty ``issues`` list;
    anything else forces another render iteration (up to
    ``MAX_CRITIC_ITERATIONS``).
    """

    model_config = {"validate_assignment": True}

    iteration: int = Field(
        ge=0,
        le=MAX_CRITIC_ITERATIONS,
        description="Critic iteration that produced this report (0 = first render).",
    )
    lufs_integrated: float = Field(description="Integrated loudness of the render in LUFS.")
    true_peak_db: float = Field(description="True peak of the render in dBTP.")
    clipping: bool = Field(description="True when any sample clips (0 dBFS reached).")
    phase_correlation: float = Field(
        ge=-1.0,
        le=1.0,
        description="Stereo phase correlation: 1.0 = mono-compatible, <0 = out of phase.",
    )
    verdict: Literal["approved", "needs_revision"] = Field(
        description="approved = ship it; needs_revision = re-render with suggested_fixes."
    )
    spectral_balance: dict[str, float] = Field(
        default_factory=dict,
        description="Per-band level in dB, e.g. {'sub': -3.1, 'presence': 1.2}.",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Blocking defects found (masking, drift, harshness, ...).",
    )
    suggested_fixes: dict[str, Any] = Field(
        default_factory=dict,
        description="Corrective DSP parameters for the next render, keyed by tool.",
    )
    summary: str = Field(default="", description="One-paragraph plain-language verdict.")

    @model_validator(mode="after")
    def _check_approved_means_clean(self) -> "CriticReport":
        if self.verdict == "approved" and self.issues:
            raise ValueError("an 'approved' report must have an empty issues list")
        return self


# ---------------------------------------------------------------------------
# LangGraph workflow state
# ---------------------------------------------------------------------------


class RemixAgentState(TypedDict):
    """State carried through the LangGraph multi-agent workflow.

    Keys start empty/``None`` and are filled in by successive agents:
    Theory/Stem Engineer fills ``bpm``/``key``/``raga``/``tala``/``stems``,
    the Director fills ``blueprint``, the Producer fills
    ``generated_stems``, the Engineer fills ``mix_params``/``master_path``/
    ``hook_path``, and the Critic fills ``critic_report`` and flips
    ``approved``.

    ``errors`` is an append-only channel (``operator.add`` reducer) so
    parallel agents accumulate diagnostics instead of overwriting each other.
    ``notes`` is the same kind of append-only channel for non-fatal
    diagnostics (tempo/key estimates, fallback notices, ...): things the
    user should see but that did not fail the run. Real failures go in
    ``errors``; informational notes go in ``notes``.
    """

    input_audio_path: str
    reference_track_path: str | None
    creative_direction: str
    bpm: float | None
    key: str | None
    scale: str | None
    raga: str | None
    tala: str | None
    stems: dict[str, str]
    blueprint: ArrangementBlueprint | None
    generated_stems: dict[str, str]
    mix_params: dict[str, Any]
    master_path: str | None
    hook_path: str | None
    critic_report: CriticReport | None
    iteration_count: int
    approved: bool
    errors: Annotated[list[str], operator.add]
    notes: Annotated[list[str], operator.add]
    sample_rate: int


def create_initial_state(
    input_audio_path: str,
    creative_direction: str,
    reference_track_path: str | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> RemixAgentState:
    """Build a fresh workflow state for one remix run.

    Raises ``ValueError`` if ``sample_rate`` is anything other than 44100 —
    the studio never resamples mid-pipeline.
    """
    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"sample_rate must be {SAMPLE_RATE} Hz studio-wide, got {sample_rate}"
        )
    return {
        "input_audio_path": input_audio_path,
        "reference_track_path": reference_track_path,
        "creative_direction": creative_direction,
        "bpm": None,
        "key": None,
        "scale": None,
        "raga": None,
        "tala": None,
        "stems": {},
        "blueprint": None,
        "generated_stems": {},
        "mix_params": {},
        "master_path": None,
        "hook_path": None,
        "critic_report": None,
        "iteration_count": 0,
        "approved": False,
        "errors": [],
        "notes": [],
        "sample_rate": sample_rate,
    }
