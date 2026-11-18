"""Single entry point for the Autonomous Multi-Agent AI Music Producer
& Remix Studio.

``python main.py --input song.wav --output out/ [--export-daw]`` runs the
full pipeline — Director -> Producer -> Engineer -> Critic with the
re-render loop — and, with ``--export-daw``, also writes a REAPER project
(.RPP) plus the canonical ``stems_and_midi/`` layout for the produced
stems. The same flow is available as a Python API via :func:`remix`.

Wiring responsibilities (this module is the wiring, the agents do the work)
-----------------------------------------------------------------------------
* **Health check.** Before the run, each Layer 3 server is probed with a
  fast ``list_tools`` call over stdio. A missing or unresponsive server is
  reported honestly in the returned ``notes`` and the run continues — the
  Producer's DSP fallback covers stems/generation, so a down server is a
  quality downgrade, never a crash.
* **MCP timeouts.** The Producer's default client budgets (600 s for
  generation, 900 s for stems) are too small for the real models on this
  CPU box: MusicGen runs ~27 s of wall clock per second of audio on
  short clips, but a full 30 s beat runs slower (~50-60 s/s on this
  throttled CPU) — the Veezha Veera 30 s beat (16 bars @ 126 BPM) hit
  the 1800 s budget and timed out, falling back to the DSP synth kick.
  BS-RoFormer needs ~19 min per 30 s of audio (~1140 s).
  :func:`apply_mcp_timeouts` raises both budgets to 3600 s at wiring
  time (a module-attribute override on ``agents.producer_agent`` — no
  producer file is changed). The 600 s budget would have timed a 30 s
  model beat out and fallen back to DSP; with 3600 s the real models
  get to finish. See ``mcp_servers/generative_audio_server.py``
  (measured 27 s/s) and ``mcp_servers/stem_theory_server.py`` for the
  rates this is sized from.
* **DAW export.** The export step calls the DAW export server's tools
  directly (plain function calls on the imported module, not MCP), stages
  the produced stems + generated stems into a scratch directory, and
  transcribes the melodic stem to MIDI on a best-effort basis. Export
  failures never fail the remix itself — they are recorded in ``errors``
  with ``daw_project_path`` left as ``None``.

Input validation (exists, 44 100 Hz, >= 30 s) happens in
``agents.graph.run_remix`` before the graph runs; invalid input raises
``FileNotFoundError`` / ``ValueError`` from the API and exits 2 from the
CLI with a one-line error, exactly like ``agents.graph``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Import shim: make `agents.*` / `core.*` / `mcp_servers.*` importable
# whether this file is run as a script (`python main.py ...`, sys.path[0]
# is the project root already) or as a module (`python -m main`).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import agents.producer_agent as producer_agent  # noqa: E402
from agents.graph import MIN_TRACK_DURATION_S, run_remix  # noqa: E402
from agents.state import SAMPLE_RATE  # noqa: E402

__all__ = [
    "remix",
    "main",
    "check_servers",
    "apply_mcp_timeouts",
    "SERVER_PATHS",
    "GENERATE_TIMEOUT_S",
    "STEM_TIMEOUT_S",
    "HEALTH_CHECK_TIMEOUT_S",
]

logger = logging.getLogger(__name__)

#: Layer 3 server scripts (primary path for the Producer).
SERVER_PATHS: dict[str, Path] = {
    "stem_theory": _PROJECT_ROOT / "mcp_servers" / "stem_theory_server.py",
    "generative": _PROJECT_ROOT / "mcp_servers" / "generative_audio_server.py",
}

#: Tools each server must expose for the health check to call it reachable.
EXPECTED_TOOLS: dict[str, set[str]] = {
    "stem_theory": {"separate_stems"},
    "generative": {"generate_beat", "generate_riser"},
}

#: Health-check budget per server (seconds). Launch + import + list_tools
#: is seconds on this box; 120 s is headroom, not an expectation.
HEALTH_CHECK_TIMEOUT_S: float = 120.0

#: Raised MCP client budgets applied at wiring time (see module docstring).
#: MusicGen ~27 s wall per second of audio on short clips, but a full
#: 30 s beat runs ~50-60 s/s on this throttled CPU: the Veezha Veera
#: 30 s beat (16 bars @ 126 BPM) exceeded the old 1800 s budget and
#: timed out. BS-RoFormer ~19 min per 30 s of audio (~1140 s). 3600 s
#: covers both with headroom; the Producer's 600 s / 900 s defaults
#: would time both out on a full-length track and fall back to DSP.
GENERATE_TIMEOUT_S: float = 3600.0
STEM_TIMEOUT_S: float = 3600.0

#: Name of the REAPER project file written into the output directory.
DAW_PROJECT_FILENAME = "remix.RPP"


# ---------------------------------------------------------------------------
# Small async plumbing (mirrors agents.producer_agent._run_async)
# ---------------------------------------------------------------------------


def _run_async(coro: Any) -> Any:
    """Drive an async coroutine from sync code.

    ``asyncio.run`` when there is no running loop; otherwise run it on a
    fresh loop in a helper thread.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        result: dict[str, Any] = {}
        error: dict[str, BaseException] = {}

        def _runner() -> None:
            try:
                result["value"] = asyncio.run(coro)
            except BaseException as exc:  # noqa: BLE001 — re-raised below
                error["error"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()
        if error:
            raise error["error"]
        return result.get("value")


async def _list_server_tools_async(server_path: Path, timeout_s: float) -> list[str]:
    """List a server's tool names over stdio (FastMCP Client).

    Raises on a missing script, a failed launch, or a timeout — the
    caller turns that into an honest "unreachable" note, never a crash.
    """
    if not server_path.is_file():
        raise FileNotFoundError(f"MCP server script not found: {server_path}")
    from fastmcp import Client

    # Pass the Path (not a string): FastMCP infers stdio transport from
    # a Path; string inference is deprecated (removed in FastMCP 5).
    async with Client(server_path, timeout=timeout_s) as client:
        tools = await client.list_tools()
    return [getattr(tool, "name", str(tool)) for tool in tools]


# ---------------------------------------------------------------------------
# Wiring: health check + timeouts
# ---------------------------------------------------------------------------


def check_servers(timeout_s: float = HEALTH_CHECK_TIMEOUT_S) -> dict[str, dict[str, Any]]:
    """Probe the Layer 3 servers and report reachability.

    Each server is launched over stdio and asked to list its tools — a
    fast check that loads no model. Returns
    ``{name: {"reachable": bool, "tools": [...], "note": str}}``.
    A missing script, a failed launch, or a timeout all yield
    ``reachable: False`` with the reason in ``note``; this function never
    raises, so a down server degrades the run to the DSP fallback instead
    of crashing it.
    """
    statuses: dict[str, dict[str, Any]] = {}
    for name, path in SERVER_PATHS.items():
        try:
            tools = _run_async(_list_server_tools_async(path, timeout_s))
        except Exception as exc:  # noqa: BLE001 — any failure = unreachable
            statuses[name] = {
                "reachable": False,
                "tools": [],
                "note": (
                    f"{name} server unreachable ({type(exc).__name__}: {exc}) — "
                    "producer will use its DSP fallback for this step"
                ),
            }
            continue
        missing = EXPECTED_TOOLS[name] - set(tools)
        if missing:
            statuses[name] = {
                "reachable": False,
                "tools": sorted(tools),
                "note": (
                    f"{name} server answered but is missing tools: "
                    f"{sorted(missing)} — producer will use its DSP "
                    "fallback for this step"
                ),
            }
        else:
            statuses[name] = {
                "reachable": True,
                "tools": sorted(tools),
                "note": f"{name} server reachable ({len(tools)} tools)",
            }
    return statuses


def apply_mcp_timeouts() -> dict[str, float]:
    """Raise the Producer's MCP client budgets for the real models.

    Sets ``agents.producer_agent.GENERATE_TIMEOUT_S`` and
    ``STEM_TIMEOUT_S`` to the values sized in the module docstring
    (3600 s each). This is a wiring-time override on the imported
    module — no producer file is changed, and the Producer reads these
    globals at call time, so the raised budgets apply to the run that
    follows. Returns the applied ``{"generate": ..., "stems": ...}``.
    """
    producer_agent.GENERATE_TIMEOUT_S = GENERATE_TIMEOUT_S
    producer_agent.STEM_TIMEOUT_S = STEM_TIMEOUT_S
    logger.info(
        "MCP timeouts raised for real models: generate=%.0fs stems=%.0fs",
        GENERATE_TIMEOUT_S,
        STEM_TIMEOUT_S,
    )
    return {"generate": GENERATE_TIMEOUT_S, "stems": STEM_TIMEOUT_S}


# ---------------------------------------------------------------------------
# DAW export (direct function calls on the imported server module)
# ---------------------------------------------------------------------------


def _stage_stems_for_export(
    stems: dict[str, str],
    generated_stems: dict[str, str],
    staging_dir: Path,
) -> tuple[list[str], list[str]]:
    """Copy the produced stems + generated stems into ``staging_dir``.

    The primary-path servers scatter their outputs (BS-RoFormer stems
    next to the input file, MusicGen output under ``outputs/generative/``)
    while the DAW export tool reads one directory — so main.py stages
    them together first. Returns ``(staged_paths, notes)``; a missing
    file is skipped with a note instead of failing the export.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []
    notes: list[str] = []
    for source in list(stems.values()) + list(generated_stems.values()):
        src = Path(source)
        if not src.is_file():
            notes.append(f"daw export: stem file missing, skipped: {source}")
            continue
        dest = staging_dir / src.name
        if src.resolve() != dest.resolve():
            shutil.copy2(str(src), str(dest))
        staged.append(str(dest))
    return staged, notes


def _transcribe_melody_midi(
    stems: dict[str, str],
    staging_dir: Path,
    bpm: float,
) -> str | None:
    """Transcribe the melodic stem to MIDI (best-effort, never fatal).

    Prefers the ``melodic`` stem, then ``harmonic``, then ``vocals``.
    Returns the .mid path or ``None`` when no suitable stem exists or
    transcription fails — the .RPP is still written without a MIDI track.
    Uses the DAW server's ``audio_to_midi`` directly (Basic Pitch when
    available, otherwise its clearly-labelled onset fallback).
    """
    from mcp_servers import daw_export_server

    for key in ("melodic", "harmonic", "vocals"):
        path = stems.get(key)
        if not path or not Path(path).is_file():
            continue
        midi_path = staging_dir / f"{Path(path).stem}.mid"
        try:
            result = daw_export_server.audio_to_midi(
                input_path=path,
                output_path=str(midi_path),
                bpm=float(bpm),
            )
            logger.info(
                "daw export: transcribed %s -> %s (%s, %d notes)",
                key,
                midi_path,
                result.get("method"),
                result.get("note_count"),
            )
            return str(midi_path)
        except Exception as exc:  # noqa: BLE001 — MIDI is optional
            logger.warning("daw export: MIDI transcription failed: %s", exc)
            return None
    return None


def _export_daw(
    final_state: dict[str, Any],
    output_dir: Path,
) -> tuple[str | None, list[str], list[str]]:
    """Write the REAPER project + stems_and_midi/ layout for a finished run.

    Returns ``(rpp_path_or_None, notes, errors)``. Needs a blueprint and
    at least one stem; anything missing is a note, never an exception
    into the caller — the remix itself already succeeded.
    """
    notes: list[str] = []
    errors: list[str] = []
    blueprint = final_state.get("blueprint")
    stems = dict(final_state.get("stems") or {})
    generated = dict(final_state.get("generated_stems") or {})
    if blueprint is None:
        notes.append("daw export skipped: no blueprint in the final state")
        return None, notes, errors
    if not stems and not generated:
        notes.append("daw export skipped: no stems in the final state")
        return None, notes, errors

    from mcp_servers import daw_export_server

    staging_dir = output_dir / "_daw_stems"
    try:
        staged, stage_notes = _stage_stems_for_export(stems, generated, staging_dir)
        notes.extend(stage_notes)
        if not staged:
            notes.append("daw export skipped: no stem files could be staged")
            return None, notes, errors
        midi_files: list[str] = []
        midi_path = _transcribe_melody_midi(stems, staging_dir, float(blueprint.bpm))
        if midi_path is not None:
            midi_files.append(midi_path)
        else:
            notes.append("daw export: no MIDI track (no melodic stem to transcribe)")
        result = daw_export_server.export_daw_project(
            stems_dir=str(staging_dir),
            blueprint_json=blueprint.model_dump(mode="json"),
            output_path=str(output_dir / DAW_PROJECT_FILENAME),
            midi_files=midi_files or None,
        )
        rpp_path = str(result["output_path"])
        notes.append(
            f"daw export: wrote {rpp_path} ({result['track_count']} tracks, "
            f"{result['midi_tracks']} MIDI) + stems_and_midi/ "
            f"({len(result.get('tracks', []))} files in manifest)"
        )
        return rpp_path, notes, errors
    except Exception as exc:  # noqa: BLE001 — export never fails the remix
        errors.append(f"daw export failed ({type(exc).__name__}: {exc})")
        return None, notes, errors
    finally:
        # The .RPP references stems_and_midi/ by relative path, so the
        # staging dir is disposable once the export has copied from it.
        shutil.rmtree(staging_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _validate_input(input_path: str) -> None:
    """Fail fast on a bad input, before the health check runs.

    Mirrors the validation in ``agents.graph.run_remix`` (which re-checks
    before the graph runs): file must exist, be a readable 44 100 Hz
    WAV, and be at least 30 s long. Same error types and messages, so a
    bad input costs milliseconds, not an 11 s server probe first.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"input audio not found: {input_path}")
    import soundfile as sf

    try:
        info = sf.info(input_path)
    except Exception as exc:  # unreadable/corrupt file: say so plainly
        raise ValueError(f"cannot read input audio {input_path!r}: {exc}") from exc
    if info.samplerate != SAMPLE_RATE:
        raise ValueError(
            f"input audio sample rate must be {SAMPLE_RATE} Hz, "
            f"got {info.samplerate} Hz ({input_path})"
        )
    duration_s = info.frames / info.samplerate
    if duration_s < MIN_TRACK_DURATION_S:
        raise ValueError(
            f"track is {duration_s:.1f} s; ArrangementBlueprint requires "
            f">= {MIN_TRACK_DURATION_S:.0f} s for the 30 s viral-hook window "
            f"(input: {input_path})"
        )


def remix(
    input_path: str,
    output_dir: str,
    creative_direction: str = "",
    target_lufs: float = -14.0,
    club_master: bool = False,
    reference_path: str | None = None,
    export_daw: bool = False,
) -> dict[str, Any]:
    """Run one full remix end to end and return a summary dict.

    Health-checks the Layer 3 servers, raises the MCP timeouts for the
    real models, runs the full graph (Director -> Producer -> Engineer ->
    Critic with the re-render loop), and when ``export_daw=True`` also
    writes the REAPER project + ``stems_and_midi/`` layout.

    Returns ``{"master_path", "hook_path", "approved", "iterations",
    "lufs", "true_peak", "daw_project_path" (or None), "notes",
    "errors"}``. ``lufs`` / ``true_peak`` are ``None`` when the run
    ended before the Critic reported (e.g. the Director failed).

    Raises ``FileNotFoundError`` when the input audio does not exist and
    ``ValueError`` for anything other than 44 100 Hz input, or for a
    track shorter than 30 s — the same validation as
    ``agents.graph.run_remix``, before the graph runs.
    """
    _validate_input(input_path)
    notes: list[str] = []
    statuses = check_servers()
    for name, status in statuses.items():
        notes.append(f"server {name}: {status['note']}")
    timeouts = apply_mcp_timeouts()
    notes.append(
        "mcp timeouts raised for real models: "
        f"generate={timeouts['generate']:.0f}s stems={timeouts['stems']:.0f}s "
        "(MusicGen ~27 s/s, BS-RoFormer ~19 min/30 s on CPU)"
    )

    final_state = run_remix(
        input_audio_path=input_path,
        output_dir=output_dir,
        creative_direction=creative_direction,
        target_lufs=target_lufs,
        club_master=club_master,
        reference_track_path=reference_path,
    )

    # Notes/errors accumulate from the graph state; the wiring notes go first.
    all_notes: list[str] = notes + list(final_state.get("notes") or [])
    all_errors: list[str] = list(final_state.get("errors") or [])

    daw_project_path: str | None = None
    if export_daw:
        daw_project_path, daw_notes, daw_errors = _export_daw(
            final_state, Path(output_dir)
        )
        all_notes.extend(daw_notes)
        all_errors.extend(daw_errors)

    critic = final_state.get("critic_report")
    return {
        "master_path": final_state.get("master_path"),
        "hook_path": final_state.get("hook_path"),
        "approved": bool(final_state.get("approved")),
        "iterations": int(final_state.get("iteration_count") or 0),
        "lufs": (float(critic.lufs_integrated) if critic is not None else None),
        "true_peak": (float(critic.true_peak_db) if critic is not None else None),
        "daw_project_path": daw_project_path,
        "notes": all_notes,
        "errors": all_errors,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python main.py --input song.wav --output out/``.

    Flags mirror ``agents.graph`` (``--input`` / ``--output`` /
    ``--direction`` / ``--target-lufs`` / ``--club`` / ``--reference``)
    plus ``--export-daw``. Exit codes: 0 = approved (and the DAW export
    succeeded when requested), 1 = not approved or the requested DAW
    export failed, 2 = invalid input (one-line error, no traceback).
    """
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
    parser.add_argument(
        "--export-daw",
        action="store_true",
        help="Also write a REAPER project (.RPP) + stems_and_midi/ layout.",
    )
    args = parser.parse_args(argv)

    try:
        result = remix(
            input_path=args.input,
            output_dir=args.output,
            creative_direction=args.direction,
            target_lufs=args.target_lufs,
            club_master=args.club,
            reference_path=args.reference,
            export_daw=args.export_daw,
        )
    except (FileNotFoundError, ValueError) as exc:
        # Input problems fail fast with one clean line, never a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"master: {result['master_path']}")
    print(f"hook: {result['hook_path']}")
    print(f"approved: {result['approved']}")
    print(f"iterations: {result['iterations']}")
    if result["lufs"] is not None:
        print(f"lufs: {result['lufs']:.2f}  true_peak: {result['true_peak']:.2f} dBTP")
    if result["daw_project_path"]:
        print(f"daw_project: {result['daw_project_path']}")
    if result["notes"]:
        print(f"notes: {result['notes']}")
    if result["errors"]:
        print(f"errors: {result['errors']}")
    if not result["approved"]:
        return 1
    if args.export_daw and result["daw_project_path"] is None:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
