# Autonomous Remix Studio — Sell Plan (draft, 2026-11-18)

Prepared after the full build finished (Round-7 verification GO, delivered
2026-11-18 ~06:59 IST). Sir asked on 2026-11-17 whether the studio can be
sold once fully built, and ordered build-first. The build is now done, so
this is the sell check he asked for first: demo, model licences, GPU cost,
competitors.

## Headline: do NOT sell the full generative remix yet

The studio's generative engine, MusicGen-Melody, is **non-commercial**.

- Confirmed in the studio's own files:
  `models/musicgen-melody/README.md` front matter says `license: cc-by-nc-4.0`,
  and its License section says: "Code is released under MIT, model weights
  are released under CC-BY-NC 4.0."
- Independent checks agree (searched 2026-11-18): Meta's AudioCraft code is
  MIT, the MusicGen weights are CC-BY-NC 4.0 — no commercial use, no selling
  MusicGen output, no client work with it.
- Consequence: any remix whose new beats/melodies came from MusicGen cannot
  be sold as a service or a product. The Layer-3 "real generative" remix is
  the blocked part.

## What IS sellable, and what is still unchecked

| Part | Status |
|---|---|
| DSP engine + mastering (LUFS -14.00, critic loop, hook cut, DAW export) | Sellable in principle — our own code |
| BS-RoFormer stem separation | **Licence unchecked** — code is MIT, but the checkpoint file `bs_roformer_4stems_ft` has no licence line in its config; a similar public checkpoint (anvuew) declares GPL-3.0. Must pin down which checkpoint this is before selling stem work |
| MusicGen beats/melodies | **Blocked** — CC-BY-NC 4.0, see above |
| Remixing a client's song | Only if the client owns the song / has the rights. A remix of someone else's copyrighted song (e.g. Veezha Veera) cannot be sold or publicly demoed |

## The two routes he named, re-priced

**Route 1 — Per-song remix service (fastest to start).**
Sellable today ONLY as a DSP + stems service with NO MusicGen audio in the
deliverable, and only for songs the client owns. That is still a real
product: real stems, rearrangement, mastered -14 LUFS, 30 s viral hook,
REAPER project. Before quoting a price: (a) settle the BS-RoFormer
checkpoint licence, (b) price GPU time — on this CPU a full song is
infeasible (stems ~19 min per 30 s, MusicGen ~35–40 s per second of audio),
so every paid job needs rented GPU time, which is not yet priced.

**Route 2 — Software subscription for producers.**
Not ready. Needs the MusicGen swap first (below), plus a website, demo,
and a way to run it on the buyer's machine or a hosted GPU. Do not price
this until Route 1 has paid jobs.

## Fix that unblocks the full remix

Swap MusicGen for a commercially-licensed generator. One candidate found in
search (2026-11-18, NOT yet verified against its model card): ACE-Step 1.5,
reported MIT weights. Verify its card, then swap it into
`mcp_servers/generative_audio_server.py` and re-run the Round-7 test before
selling anything generative.

## Competitors (checked 2026-11-18, third-party comparisons)

- **LALAL.AI** — Lite $7.50/mo, Pro $15/mo (annual billing); stems only.
- **Moises** — Premium ~$3.99–$5.99/mo, Pro ~$9.99–$29.99/mo (varies by
  store/region); stems + practice tools.
- Lesson: stems alone are a cheap commodity. Our edge is NOT stems — it is
  the full remix (arrangement + new beats + mastered hook + DAW project).
  Price the service as a remix, not as stem separation.

## Demo assets already on disk (private proof, not public demos)

- Real-model 30 s test: `outputs/r7/main_e2e/final_master_full.wav` +
  `viral_hook_30s.wav` + `remix.RPP` (approved iteration 0, LUFS -14.00).
- Full-song Veezha Veera remix: `outputs/yt-test/d12-rerun/final_master_full.wav`
  + `viral_hook_30s.wav` — **copyrighted song, private listening only**;
  do not post it or use it as a public demo.

## Decision for sir (one tap each)

1. **Route 1 now, DSP-only** — I settle the BS-RoFormer licence + GPU price,
   then bring one per-song price to approve.
2. **Swap the generator first** — replace MusicGen with a commercial-licensed
   model, re-verify, then sell the full remix.
3. **Pause selling** — keep the studio as his own tool for now.

No money spent, nothing sent to anyone, no public demo posted.
