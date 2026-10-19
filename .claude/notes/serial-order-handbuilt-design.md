# Serial order: the hand-built competitive-queuing design

Reference solution for the `serial-order` task, saved as `neat-dnfs/tests/fixtures/serial-order-handbuilt.dnf`. Under
the evaluator of 19 October 2026 it scores 0.994, with every recall term at 1.0, in 50 of 50 evaluations. That
evaluator has an evenly spaced demonstration (`cbbf9f6`) and requires each peak to rise only once the previous one has
gone (`51d03d9`). It still scores 0.994 once each peak's credit is weighted by its height (`f8a1098`), since its
peaks reach about 11 (`serial-order-peak-height-credit.md`).

Every setting is inside the evolvable ranges, and every connection is one evolution can make: none leaves the output
field (`output-field-cannot-be-a-connection-source.md`). Two tests guard this: `SerialOrder's evolvable ranges reach
every setting of the hand-built design` and `SerialOrder's hand-built design uses only connections evolution can make`.

The serial-order hand-off bundle (`.claude/temp/serial-order-and-timing-experiment/`, gitignored scratch) holds:
- a dnf-composer replay of one evaluation, at
  `serial-order/architectures/reachable/serial-order-handbuilt-reachable-replay.dnf` (gap 700, cycles 1000, 800,
  1500; X at t = 1550, 2550, 3350), with replays at gaps 300 and 1500 beside it;
- the earlier design, whose output did the competition and fed the done field, at
  `serial-order/architectures/evenly-spaced/handbuilt-evenly-spaced-replay.dnf`;
- the builder that moved the competition into a hidden field, `tools/handbuilt/order_reachable.py`;
- the numpy prototype and exporter, `tools/handbuilt/order_design.py` and `order_export.py`. They still write the
  original, out-of-range version.

## How it works

Order is stored as a primacy gradient and read out by competition, never by releasing the store, so it can be
recalled again on every trigger.

| Field | Role |
|---|---|
| nf 1 input I | Events as bumps; X as a boost of the whole field |
| nf 3 memory M | Holds each event as a self-sustained bump. I->M has local excitation and global inhibition that cancel for the boost, so X is never stored |
| nf 4 clamp C | Uniform; on as soon as M holds anything (M->C global), and stays on while it does |
| nf 5 integrator G | tau 1e5. M drives each stored position up at rate s/tau, and C inhibits all of G at the same rate, so each trace freezes when its event is stored: D highest, then C, A, B. That gradient is the order |
| nf 6 recall latch R | Uniform; latches on when the X boost arrives while C is on, so the demonstration's X never sets it |
| nf 8 selection S | R + M + G input, strong global inhibition: selects the stored item with the highest G level |
| nf 7 done D | Integrates the winner from S (tau 75), ignites and suppresses it in S, so the next-highest item wins. Every X boost clears it (I->D global inhibition), so the sequence starts again |
| nf 2 output O | Copies S. It keeps the settings a new output field starts with in this task |

In dnf-composer, each recall cycle shows D, C, A, B one at a time in the output as full peaks (amplitude 11.0-12.4),
about 120 apart, each about 10 after the same peak in S. No two are ever up at once: 8-34 pass between one peak going
and the next rising. The output is silent again about 590 after each X, and stays at most -9.6 during the
demonstration. Replays at gaps 300, 700 and 1500 all look the same.

## Parameters

| Element | Setting |
|---|---|
| nf 1 | tau 10, h -10, gk(2, 3) |
| nf 2 | tau 10, h -10, gk(2, 8, global -0.01): the starting output of `templates/serial-order.dnf` |
| nf 3 | tau 10, h -5, mhk(2.5, 20, 6, 12) |
| nf 4 | tau 10, h -5, gk(2, 6) |
| nf 5 | tau 1e5, h +0.6, no self-excitation |
| nf 6 | tau 10, h -15, gk(2, 15) |
| nf 7 | tau 75, h -5, mhk(2.5, 20, 6, 12) |
| nf 8 | tau 10, h -15, gk(2.5, 15, global -1.0) |
| I->M | gk(2, 20, global -0.1) |
| M->C | gk(2, 1, global +0.5) |
| M->G, C->G | gk(2, +220), gk(2, -220) |
| I->R, C->R | gk(2, 1, global +0.1), gk(2, 5) |
| R->S, M->S, G->S | gk(2, 10), gk(2, 5), gk(2, 8) |
| S->D, D->S | gk(2, 8), gk(3, -30) |
| S->O | gk(2, 15) |
| I->D | gk(2, -1, global -0.2) |

## The competition moved out of the output (19 October 2026)

The earlier design did the competition in the output itself, and the done field read the winner from there
(`gk cg 2-7`). Evolution can never make a connection out of the output. So the output's settings and connections moved
unchanged to a new hidden field, the selection field S (nf 8). The done field now reads from S and suppresses S. The
output became a follower: one connection S->O, and the settings a new output field starts with. S behaves exactly as
the old output did, and the score is unchanged at 0.994.

How strong S->O must be, with the output otherwise untouched (bundle `serial-order/architectures/reachable/`):
- At width 2, 12-18 scores 0.994 in 50 of 50 evaluations; 15 is in the middle. 10 never lifts the output (0.341). 20
  loses a little one-at-a-time credit in 5 of 50 evaluations (min 0.988), and 30 more (0.979).
- A new connection draws a random sign, a width from 1-10 and an amplitude log-uniformly from 1-1000
  (`ConnectionGene::initializeKernel`). Every width has a working band, starting at a threshold that rises with the
  width: 10 at width 1, 12 at 2, 18 at 4, 25 at 6 and 30 at 8, and above 30 at width 10 (`sweep2-copy-width.txt`, 10
  evaluations each). Below it the output stays silent (0.33). Above it the score falls slowly, to 0.94-0.98 at 100
  for widths up to 4.

A sweep over the output's own settings (`sweep1.txt`, copy width 2, 10 evaluations each, script
`order_reachable_sweep.py`) puts the follower's working range at:
- **Without self-excitation:** every resting level from -15 to -1 works once S->O reaches its threshold: 3 at -1,
  20 at -15. A stronger copy than needed costs one-at-a-time credit gradually, down to 0.86-0.98 at 100.
- **With the starting self-excitation (gk 8):** about the same thresholds up to a resting level of -7.5, but a strong
  copy costs credit sooner. From -5 up, peaks outlast S (0.937) or the output ignites (0.24-0.38).

In the sections below, O means the field that did the competition: the output then, S now.

## Retuned for the evenly spaced demonstration (19 October 2026)

The demonstration used to put D, C, A, B at 100, 600, 800, 1800 after X. It now puts them 200 apart, at 200, 400,
600, 800. The design scored 0.73 under the new schedule (f5 0.34, f6 0.6), because neighbouring items tied. Six
settings changed:

| Setting | Was | Now | Why |
|---|---|---|---|
| M->G, C->G | +110, -110 | +220, -220 | Each gradient step is s/tau x the interval. At 200 apart the steps were 0.22 everywhere, the old design's narrowest (C to A); doubling s gives 0.44 |
| G->O | 6 | 8 | Wider differences between the items' inputs to the output |
| h(O) | -13 | -15 | Silence at the demonstration's X (below) |
| M->O | 6 | 5 | Moves lift from the latch to memory (below) |
| R->O | 9 | 10 | Keeps the lowest item, B, able to rise |

All six stay inside the evolvable ranges; h(O) -15 is the range minimum.

What the field probes showed, step by step (bundle `tools/evaluator-diagnostics/diag_field_probe.cpp.txt`):
- **The first trigger lifts all items at once.** The latch switches the output's lift on within a few steps. Every
  stored item jumps above threshold, and the competition only separates two of them if their inputs differ by about
  2 or more. In the old schedule the first pair, D and C, differed by 2.2. With equal spacing and the old gains it was
  1.1, so D and C rose together. Later triggers release the items gradually through the done field, so they tolerate
  smaller differences.
- **The demonstration's X transiently opens the latch.** The boost drives R to about 20 for a moment, even with memory
  empty. That lifts the whole output by h(O) + R->O + G->O x sigmoid(h(G)), and the output's global inhibition only
  holds a uniform field below threshold while that sum is small. Stronger R->O and G->O made the output brush
  threshold there. M->O is zero at that moment, since nothing is stored yet. So lowering h(O) and raising M->O by the
  same amount keeps recall unchanged and restores the margin.
- **The lift on stored items must be balanced.** h(O) + M->O + R->O has to sit between about 0 and +0.5. Below, B
  never rises. Above, the first two items tie.

Sweeps (bundle `serial-order/architectures/evenly-spaced/sweep*.txt`, 10 evaluations each, script
`tools/inrange/evenly_spaced_variant.py`):
- 49 of 81 variants score 0.99 or more, over h(O) -15 to -14, M->O 4-6, G->O 7-9 and that lift sum 0 to +1, with
  s 220.
- G->O 7, 8 and 9 all pass 50 of 50 evaluations.

## Inside the evolvable ranges (18 October 2026, old demonstration)

Scores in this section are under the old demonstration (onsets 100, 600, 800, 1800).

The first version had seven settings evolution could not reach. Five were outside the ranges and two were in range but
off the 0.1 grid that a connection's global amplitude moves on (it starts at 0). Each was fixed in the fixture where an
equivalent in-range value works, and in `config/solutions/serial-order.json` where none does. The test
`SerialOrder's evolvable ranges reach every setting of the hand-built design` guards this.

| Setting | Was | Now | Where |
|---|---|---|---|
| Local amplitude of M->C, I->R, I->D | 0 | +1, +1, -1 | fixture; connection amplitudes are 1 ... 1000 |
| M->C global | +1.0 | +0.5 | fixture; +0.5 is the connection maximum and works as well |
| I->M | gk(10, -0.05) | gk(20, -0.1) | fixture; same ratio, so the boost still cancels |
| I->R global | +0.04 | +0.1, with h(R) -10 -> -15 | fixture; +0.1 alone (0.87) or +0.05 (0.86) lets events open the latch |
| h(G) | +0.6 | unchanged | config: resting-level maximum -0.5 -> +1.0 |
| O kernel global | -1.2 | -1.0 | fixture, and config: field Gauss-kernel global minimum -0.2 -> -1.0 |

What could not be moved into range, from variants scored in the evaluator (10 evaluations each):
- h(G) must be about +0.3 to +0.7. It scores 0.84 at 0.0, 0.55 at -0.5 (0.81-0.83 even with G->O raised to 30-60),
  and 0.85 at +1.0. The traces must sit in the sigmoid's sensitive range for the gradient to rank all four items.
- O's global inhibition must be -0.8 or stronger. It scores 0.87 at -0.6 and 0.74 at -0.2, where items rise together.

## Failure modes found while tuning

- The weakest output peak (B, lowest in the gradient) must still drive the done field to ignition, or B stays on
  and wins the start of the next cycle. O->D of 6.5 fails; 8 works.
- A slower done field (tau 80+ with I->D global -0.1) is not cleared within the 50-unit boost; I->D -0.2 fixes it.
- Two neighbouring items whose inputs to the output differ by less than about 2 rise together at the first trigger.
- Weaker competition (output global -0.6), a weaker done field (D->O -10) and a slower output (tau 20) let each peak
  rise 30-40 before the previous one has gone. That is now the fixture `serial-order-overlapping-peaks.dnf`.
