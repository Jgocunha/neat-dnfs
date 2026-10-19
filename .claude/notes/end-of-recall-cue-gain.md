# What a cue at the end of recall would add

Question asked on 19 October 2026: would serial order or sequence timing be easier to evolve from scratch if a second
cue, Y, signalled that recall has stopped? Measured answer: no.

## What Y could do

At the end of a recall it can only silence the output. In the evaluator that touches one thing: each cycle's
`lastPeakDecay`, the share of f6 for the last event's peak dying out (1 of f6's 5 parts; f6 weighs 0.19). It cannot
help:
- isolation, which is measured 50 after each onset, long before Y;
- order credit (f5);
- the missing mechanisms themselves:
  - serial order needs a ranking of the stored items and per-item suppression;
  - sequence timing needs the memory -> slow integrator -> clamp -> release-on-X chain and per-peak switch-off.

In serial order X already clears the done field of the hand-built design, so resetting between cycles is covered too.

## Measured on the from-scratch champions

These are the champions' own traces, scored under the evaluators of their runs, before serial order's demonstration
became evenly spaced (`cbbf9f6`). The argument does not depend on the schedule: Y can only touch `lastPeakDecay`.

`lastPeakDecay` was set to 1 in every cycle of each champion's own evaluation trace (`overview.jsonl`). The trace's f6
was then rebuilt as probe silence x mean over cycles of (gate x (isolation sum + lastPeakDecay) / 5); the rebuild
matched the reported f6 for all 22 champions.

| Champions | Fitness gain from a perfect Y |
|---|---|
| Sequence timing runs 24-32 | 0 |
| Sequence timing run 33 | +0.038 |
| Serial order batch 1 run 1 | +0.038 |
| Serial order batch 2 run 8 | 0, or +0.011 if Y earned credit even when B never rose |
| The other 9 serial-order champions | 0 |

The bursts already let their peaks die before the cycle ends, so a perfect Y gives 19 of 22 champions nothing. The
hand-built designs switch every peak off themselves and gain nothing either.

## Costs

Y must be told apart from X:
- a second field-wide boost is the same stimulus as X;
- a Gaussian pulse on the input field is stored like an event and recalled off the event positions, which the
  activity-at-events gate penalises;
- so it would need its own input field: a new task variant with a different topology.

The analysis script is `.claude/temp/y-trigger-analysis/y_gain.py` (gitignored scratch). It takes the hand-off
bundle's path as its argument.
