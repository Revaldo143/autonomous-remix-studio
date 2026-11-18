# Autonomous Multi-Agent AI Music Producer & Remix Studio

An autonomous pipeline that takes a song + a creative direction (or a
reference track) and produces a mastered remix, viral-hook cuts, and
DAW-compatible stems/MIDI — with a self-critique loop that re-renders
until the mix passes (max 3 iterations).

## Architecture

```
input.wav ──► [Stem/Theory] ──► [A&R Director] ──► ArrangementBlueprint
                     │                 │
                     ▼                 ▼
              [Generative Audio]  [DSP Mixing/Mastering]
                     │                 │
                     └──────► [Critic] ──► approved? ──► final_master_full.wav
                                   │  no (≤3 iters)        viral_hook_30s.wav
                                   └─► re-render ──┘       stems_and_midi/
```

* **Orchestration:** LangGraph stateful workflow (`agents/state.py` holds
  the `RemixAgentState` TypedDict and the `ArrangementBlueprint` Pydantic
  contract — all timestamps in integer milliseconds, sections tiling the
  track with no gaps or overlaps, sample rate fixed at 44 100 Hz).
* **Tools:** one FastMCP server per capability family in `mcp_servers/`
  (stems/theory, generative audio, DSP/mastering, DAW export).
* **DSP core:** `core/dsp_utils.py` — sample-accurate beat-grid alignment,
  kick-triggered sidechain compression, dynamic vocal frequency carving.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate

# 1. Torch with CUDA first (otherwise pip installs the CPU-only wheel):
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu128

# 2. Everything else:
pip install -r requirements.txt

# 3. The rubberband CLI binary (needed by pyrubberband at runtime):
#    Debian/Ubuntu: sudo apt install rubberband-cli
#    macOS:         brew install rubberband
```

## Verify the schema layer

```bash
python -c "from agents.state import ArrangementBlueprint, RemixAgentState; print(ArrangementBlueprint.model_json_schema().keys())"
```

## Run the DSP server standalone

The mastering server exposes the DSP functions as MCP tools over stdio:

```bash
python -m mcp_servers.dsp_mastering_server
```

Point any MCP client (or the LangGraph Engineer agent) at that command to
call the tools: beat-grid alignment, sidechain compression, vocal carving,
and Matchering-based reference matching.

## VRAM note (16 GB consumer GPU / Colab T4)

Heavy models (BS-RoFormer, WhisperX, MusicGen, Seed-VC) are **never held
in VRAM together**. Each MCP server loads its model on demand, runs, then
unloads it (`torch.cuda.empty_cache()`) before the next stage starts — see
`core/config.py`. A 16 GB card (or a Colab T4) runs the full pipeline
sequentially; an A100 just runs it faster. If you hit OOM, lower the
generation batch to a single stem per call before touching anything else.
```

## Exports

Every run writes, non-destructively:

| File | Spec |
| --- | --- |
| `final_master_full.wav` | 24-bit, −14 LUFS (streaming) / −9 LUFS club master |
| `viral_hook_30s.wav` | 30 s retention cut for Reels/Shorts (15 s variant too) |
| `stems_and_midi/` | isolated + generated WAV stems and `.mid` files for DAW import |
