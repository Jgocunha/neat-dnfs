# Serial order: the hand-built competitive-queuing design

Reference solution for the `serial-order` task, saved as `neat-dnfs/tests/fixtures/serial-order-handbuilt.dnf`.
It scores 0.993 in the evaluator (every recall term 1.0). A dnf-composer replay of one evaluation (gap 700, cycles
1000, 800, 1500; X at t = 2550, 3550, 4350) was written to `.claude/temp/analysis/serial-order-handbuilt-replay.dnf`.
The numpy prototype and exporter are `.claude/temp/handbuilt/order_design.py` and `order_export.py` (gitignored scratch).

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
| nf 2 | tau 10, h -13, gk(2.5, 15, global -1.2) |
| nf 3 | tau 10, h -5, mhk(2.5, 20, 6, 12) |
| nf 4 | tau 10, h -5, gk(2, 6) |
| nf 5 | tau 1e5, h +0.6, no self-excitation |
| nf 6 | tau 10, h -10, gk(2, 15) |
| nf 7 | tau 75, h -5, mhk(2.5, 20, 6, 12) |
| I->M | gk(2, 10, global -0.05) |
| M->C | global +1.0 |
| M->G, C->G | gk(2, +110), gk(2, -110) |
| I->R, C->R | global +0.04, gk(2, 5) |
| R->O, M->O, G->O | gk(2, 9), gk(2, 6), gk(2, 6) |
| O->D, D->O | gk(2, 8), gk(3, -30) |
| I->D | global -0.2 |

Robust in the prototype across gaps 300-1500 and cycle lengths 800-1500. It still works with tau(D) 70-85, O->D
7.5-8.5, G->O 5-7 and h(G) 0.5-0.7. Below tau(D) 65 the items come under 100 apart, so order credit drops slightly.

## Outside the evolvable ranges (serial-order config)

- h(G) = +0.6; resting levels are limited to -15 ... -0.5
- O's own kernel global -1.2; field kernels are limited to -0.2 ... 0
- M->C global +1.0; connection globals are limited to -2.5 ... 0.5
- M->C and I->R have local amplitude 0; Gauss connection amplitudes are limited to 1 ... 1000

So evolution cannot reach this exact design as configured. A seed built from it needs these moved into range first,
as was done for the boost timing design (v2).

## Failure modes found while tuning

- The weakest output peak (B, lowest in the gradient) must still drive the done field to ignition, or B stays on
  and wins the start of the next cycle. O->D of 6.5 fails; 8 works.
- A slower done field (tau 80+ with I->D global -0.1) is not cleared within the 50-unit boost; I->D -0.2 fixes it.
- C and A are the closest pair in the gradient (200 apart in the demonstration); with a shallow gradient (s 80) or
  stronger output global inhibition (-1.6) they rise together.
