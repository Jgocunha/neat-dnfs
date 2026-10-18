# Serial order: the hand-built competitive-queuing design

Reference solution for the `serial-order` task, saved as `neat-dnfs/tests/fixtures/serial-order-handbuilt.dnf`.
It scores 0.993 in the evaluator (every recall term 1.0). A dnf-composer replay of one evaluation (gap 700, cycles
1000, 800, 1500; X at t = 2550, 3550, 4350) is `.claude/temp/analysis/serial-order-handbuilt-inrange-replay.dnf`;
`serial-order-handbuilt-replay.dnf` beside it replays the original, pre-range-fix version. The numpy prototype and
exporter are `.claude/temp/handbuilt/order_design.py` and `order_export.py` (gitignored scratch). They still write the
original version; `.claude/temp/inrange/variant.py` applies the in-range edits to the fixture and scores variants.

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
| nf 2 output O | R + M + G input, strong global inhibition: selects the stored item with the highest G level |
| nf 7 done D | Integrates the winner (tau 75), ignites and suppresses it, so the next-highest item wins. Every X boost clears it (I->D global inhibition), so the sequence starts again |

In dnf-composer each recall cycle shows D, C, A, B one at a time as full peaks (amplitude 6-10.7, width 6), about
110 apart, with the output silent again about 550 after each X.

## Parameters

| Element | Setting |
|---|---|
| nf 1 | tau 10, h -10, gk(2, 3) |
| nf 2 | tau 10, h -13, gk(2.5, 15, global -1.0) |
| nf 3 | tau 10, h -5, mhk(2.5, 20, 6, 12) |
| nf 4 | tau 10, h -5, gk(2, 6) |
| nf 5 | tau 1e5, h +0.6, no self-excitation |
| nf 6 | tau 10, h -15, gk(2, 15) |
| nf 7 | tau 75, h -5, mhk(2.5, 20, 6, 12) |
| I->M | gk(2, 20, global -0.1) |
| M->C | gk(2, 1, global +0.5) |
| M->G, C->G | gk(2, +110), gk(2, -110) |
| I->R, C->R | gk(2, 1, global +0.1), gk(2, 5) |
| R->O, M->O, G->O | gk(2, 9), gk(2, 6), gk(2, 6) |
| O->D, D->O | gk(2, 8), gk(3, -30) |
| I->D | gk(2, -1, global -0.2) |

Robust in the prototype across gaps 300-1500 and cycle lengths 800-1500. It still works with tau(D) 70-85, O->D
7.5-8.5, G->O 5-7 and h(G) 0.5-0.7. Below tau(D) 65 the items come under 100 apart, so order credit drops slightly.

## Inside the evolvable ranges (since 18 October 2026)

The first version had seven settings evolution could not reach. Five were outside the ranges and two were in range but
off the 0.1 grid that a connection's global amplitude moves on (it starts at 0). Each was fixed in the fixture where an
equivalent in-range value works, and in `config/solutions/serial-order.json` where none does. The fixture still scores
0.993 with every recall term 1.0, and the dnf-composer replay shows the same one-at-a-time recall, with peaks of
amplitude 7-11.7. The test `SerialOrder's evolvable ranges reach every setting of the hand-built design` guards this.

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
- C and A are the closest pair in the gradient (200 apart in the demonstration); with a shallow gradient (s 80) or
  stronger output global inhibition (-1.6) they rise together.
