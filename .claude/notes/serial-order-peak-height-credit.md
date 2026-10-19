# Serial order: recall credit weighted by peak height

Since `f8a1098` (19 October 2026), each recalled event's share of serial order's recall terms (f4 recalled events,
f5 order, f6 one at a time) is `min(1, peak height / 2)`. The height is the highest activation of the event's peak
from the step it rose to the end of the cycle (`SequenceTiming::peakHeightCredit`; the trace records `peakHeight`
and `peakHeightCredit` per event). An order pair counts the smaller credit of its two events, and the last peak's
decay counts B's. Sequence timing is unchanged.

## Why

dnf-composer counts any activation above 0.00001 as a bump (`activationThreshold` in its `neural_field.cpp`), and the
evaluator never asked how tall a peak was. In the from-scratch output-feedback batch of 19 October 2026, the best
runs used that:
- run 2's output sat about 0.1 below threshold everywhere and crossed it by 0.0-0.2 at the event positions, earning
  0.92 of the recalled-event credit (re-evaluated 0.64, reported 0.80);
- run 6's two faint pairs of peaks (height 0.3-0.6) re-evaluated at 0.74;
- such flickers are noise-driven, so reported bests far exceeded re-evaluated ones.

The weight is graded, not a cutoff, so a population of faint bursts still has a gradient towards taller peaks.
The silence terms needed no change: `Solution::noBumps` already scores `exp(-height)`, so a flicker of 0.1 keeps
0.9 of its silence credit and a real peak almost none.

## Effect, 10 evaluations each

| Genome | Before | After |
|---|---|---|
| hand-built design (peaks about 11) | 0.994 | 0.994 |
| run 2, threshold flicker (fixture `serial-order-threshold-flicker.dnf`) | 0.629 | 0.338 |
| run 3, D 2.4 then B 1.3 (fixture `serial-order-low-peaks.dnf`) | 0.665 | 0.494 |
| run 6, faint pairs | 0.770 | 0.413 |
| run 1, faint burst | 0.590 | 0.436 |
| seeds no-done, no-gradient, no-latch | 0.358, 0.824, 0.329 | 0.358, 0.827, 0.329 |
| sequence-timing-boost run 18 (other task) | 0.984 | 0.984 |

An output that never peaks scores about 0.33, so the flicker now earns nothing beyond silence. Scores from before
and after `f8a1098` are not comparable.
