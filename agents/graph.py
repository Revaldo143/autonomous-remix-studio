"""LangGraph wiring for the Autonomous Multi-Agent AI Music Producer & Remix Studio.

The workflow is a four-node state machine over :class:`RemixAgentState`:

    START -> director -> producer -> engineer -> critic
                  |                         |  (route_after_critic)
                  | (no blueprint)          +-- "re_render" -> engineer
                  +-- "done" -> END         +-- "done" -------> END

``run_remix`` validates the input before the graph runs (file exists,
44 100 Hz, >= 30 s) so a bad input fails fast with one clear message.
The director never raises: on failure it returns ``errors`` and no
blueprint, and the graph routes straight to END instead of marching the
producer/engineer through wasted work and dying in the critic.

* **director** (A&R Creative Director) — analyses the track and writes the
  strict JSON :class:`ArrangementBlueprint` (sections, viral hooks, drops).
* **producer** (Music Theory / Stem / Generative) — separates stems, extracts
  BPM/key/raga and synthesises new stems. Falls back to DSP-only mode when
  the heavy MCP servers are not available.
* **engineer** (Surgical DSP Mixing & Mastering) — renders the mix through
  the DSP core and masters it to the target LUFS.
* **critic** (Multimodal RLHF Audio Critic) — inspects the render and either
  approves it or sends corrective parameters back to the engineer. The loop
  is capped at ``MAX_CRITIC_ITERATIONS`` (3) by the critic node itself:
  once the cap is hit, ``route_after_critic`` returns ``"done"`` even when
  the verdict is ``"needs_revision"``, and the run ends unapproved.

This module only wires the graph; the node implementations live in
``agents/director_agent.py`` etc. and are imported, never re-implemented.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

import soundfile as sf

# Allow running as a script (``python agents/graph.py ...``) as well as a
# module (``python -m agents.graph``) from the project root.
if __package__ in (None, ""):  # pragma: no cover - script entry branch
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.critic_agent import critic_node, route_after_critic
from agents.director_agent import director_node
from agents.engineer_agent import engineer_node
from agents.producer_agent import producer_node
from agents.state import SAMPLE_RATE, RemixAgentState, create_initial_state

__all__ = ["build_graph", "run_remix", "main"]

logger = logging.getLogger(__name__)

#: Minimum track length in seconds. The blueprint requires a 30 s viral-hook
#: window inside the track, so a shorter track can never get a valid
#: arrangement. Validated in :func:`run_remix` before the graph runs, so a
#: short track fails fast with a clear message instead of marching through
#: the producer/engineer and dying in the critic with a cryptic error.
MIN_TRACK_DURATION_S: float = 30.0


def route_after_director(state: RemixAgentState) -> str:
    """Route after the director: proceed only with a valid blueprint.

    The director never raises — on failure it returns ``errors`` and no
    blueprint. Without this guard the graph would march the producer and
    engineer through wasted work and then die in the critic
    (``critic_node needs state['master_path']``) with no usable message.
    Routing to ``"done"`` ends the run cleanly with the director's error
    in the final state instead.
    """
    return "producer" if state.get("blueprint") is not None else "done"


def build_graph() -> CompiledStateGraph:
    """Build and compile the remix workflow graph.

    Returns the compiled graph. Node functions are imported from the
    sibling agent modules; a missing module fails loudly at import time
    rather than producing a silently incomplete graph.
    """
    graph: StateGraph = StateGraph(RemixAgentState)
    graph.add_node("director", director_node)
    graph.add_node("producer", producer_node)
    graph.add_node("engineer", engineer_node)
    graph.add_node("critic", critic_node)

    graph.add_edge(START, "director")
    graph.add_conditional_edges(
        "director",
        route_after_director,
        {
            "producer": "producer",
            "done": END,
        },
    )
    graph.add_edge("producer", "engineer")
    graph.add_edge("engineer", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "re_render": "engineer",
            "done": END,
        },
    )
    return graph.compile()


def run_remix(
    input_audio_path: str,
    output_dir: str,
    creative_direction: str = "",
    target_lufs: float = -14.0,
    club_master: bool = False,
    reference_track_path: str | None = None,
) -> dict:
    """Run one full remix end to end and return the final state dict.

    Builds the initial state via :func:`create_initial_state`, seeds
    ``mix_params`` with the mastering targets and output directory, invokes
    the compiled graph, and returns the final state (which carries the
    blueprint, ``master_path``, ``critic_report``, ``iteration_count`` and
    ``approved``).

    Raises ``FileNotFoundError`` when the input audio does not exist and
    ``ValueError`` for anything other than 44 100 Hz input, or for a track
    shorter than 30 s (the blueprint needs a 30 s viral-hook window inside
    the track). Both checks happen before the graph runs, so the failure is
    a single clear message instead of a traceback from deep in the graph.
    ``club_master=True`` selects the -9 LUFS club target over
    ``target_lufs``.
    """
    if not os.path.isfile(input_audio_path):
        raise FileNotFoundError(f"input audio not found: {input_audio_path}")
    try:
        info = sf.info(input_audio_path)
    except Exception as exc:  # unreadable/corrupt file: say so plainly
        raise ValueError(f"cannot read input audio {input_audio_path!r}: {exc}") from exc
    if info.samplerate != SAMPLE_RATE:
        raise ValueError(
            f"input audio sample rate must be {SAMPLE_RATE} Hz, "
            f"got {info.samplerate} Hz ({input_audio_path})"
        )
    duration_s = info.frames / info.samplerate
    if duration_s < MIN_TRACK_DURATION_S:
        raise ValueError(
            f"track is {duration_s:.1f} s; ArrangementBlueprint requires "
            f">= {MIN_TRACK_DURATION_S:.0f} s for the 30 s viral-hook window "
            f"(input: {input_audio_path})"
        )
    os.makedirs(output_dir, exist_ok=True)

    effective_target = -9.0 if club_master else target_lufs
    state = create_initial_state(
        input_audio_path=input_audio_path,
        creative_direction=creative_direction,
        reference_track_path=reference_track_path,
    )
    state["mix_params"] = {
        "target_lufs": effective_target,
        "club_master": club_master,
        "output_dir": os.path.abspath(output_dir),
    }

    logger.info(
        "starting remix: input=%s output_dir=%s target_lufs=%.1f club=%s",
        input_audio_path,
        output_dir,
        effective_target,
        club_master,
    )
    compiled = build_graph()
    final_state = compiled.invoke(state)
    logger.info(
        "remix finished: approved=%s iterations=%d master=%s",
        final_state.get("approved"),
        final_state.get("iteration_count"),
        final_state.get("master_path"),
    )
    return dict(final_state)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m agents.graph --input song.wav --output out/``."""
    parser = argparse.ArgumentParser(
        description="Autonomous Remix Studio — run one remix end to end."
    )
    parser.add_argument("--input", required=True, help="Input audio file (44.1 kHz WAV).")
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for the master, hook and stems.",
    )
    parser.add_argument(
        "--direction",
        default="",
        help="Creative direction for the Director agent (e.g. 'kuthu + phonk').",
    )
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=-14.0,
        help="Streaming loudness target in LUFS (default -14.0).",
    )
    parser.add_argument(
        "--club",
        action="store_true",
        help="Club master: use the -9 LUFS target instead of --target-lufs.",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Optional reference track for loudness/EQ matching.",
    )
    args = parser.parse_args(argv)

    try:
        final_state = run_remix(
            input_audio_path=args.input,
            output_dir=args.output,
            creative_direction=args.direction,
            target_lufs=args.target_lufs,
            club_master=args.club,
            reference_track_path=args.reference,
        )
    except (FileNotFoundError, ValueError) as exc:
        # Input problems fail fast with one clean line, never a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2

    critic = final_state.get("critic_report")
    verdict = critic.verdict if critic is not None else "no critic report"
    print(f"master: {final_state.get('master_path')}")
    print(f"hook: {final_state.get('hook_path')}")
    print(f"approved: {final_state.get('approved')} (verdict: {verdict})")
    print(f"iterations: {final_state.get('iteration_count')}")
    if critic is not None:
        print(f"lufs: {critic.lufs_integrated:.2f}  true_peak: {critic.true_peak_db:.2f} dBTP")
    if final_state.get("notes"):
        print(f"notes: {final_state['notes']}")
    if final_state.get("errors"):
        print(f"errors: {final_state['errors']}")
    return 0 if final_state.get("approved") else 1


if __name__ == "__main__":
    raise SystemExit(main())
