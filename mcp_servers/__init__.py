"""MCP servers for the Autonomous Remix Studio.

Each server exposes one capability family as FastMCP tools:

* ``stem_theory_server`` — BS-RoFormer stems, WhisperX, Essentia, Basic Pitch.
* ``generative_audio_server`` — Stable Audio, MusicGen-Melody, Seed-VC, AudioLDM 2.
* ``dsp_mastering_server`` — Pedalboard DSP, dynamic EQ, sidechain, Matchering.
* ``daw_export_server`` — multi-track stems + PrettyMIDI project export.
"""
