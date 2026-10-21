# How serial-order-fitness-test valued the release of a recalled peak

This records the fixed-time evaluator of `f7caf34`. Recall has since been scored by order (`serial-order-fitness-test.md`,
"Recall is scored by order, not at fixed times"), and the valley below is gone.

20 October 2026, evaluator `f7caf34`, weights 0.05/0.05/0.10/0.35/0.20/0.20/0.05.

## The release in the reference

The reference releases each recalled peak through its done field, nf 7:
- the output drives nf 7 through `gk cg 2-7` at +8;
- nf 7 inhibits the output back through `gk cg 7-2` at -30;
- nf 7 has tau 75.

The fitness values that release through the recall terms:
- f4 pays one peak at the event due, at fixed times after X: 150, 290, 400 and 500;
- f6 pays a trough at each event recalled earlier in the cycle, full at 2 below the output's median;
- f7 pays troughs at 600 after X and at the cycle's end.

## What weakening the release does

The variants below change only the done loop. They were scored with `neat-dnfs-sol-eval`; the variants are in
`.claude/temp/param-ranges/`.

| Reference variant | Fitness | f4 | f5 | f6 | f7 |
|---|---|---|---|---|---|
| intact (7-2 at -30) | 0.972 | 0.92 | 1.00 | 1.00 | 1.00 |
| 7-2 at -20 | 0.975 | 0.93 | 1.00 | 1.00 | 1.00 |
| 7-2 at -12 | 0.56-0.59 | 0.40-0.47 | 1.00 | 0.04 | 0.30-0.33 |
| no release (7-2 at -8 or weaker, or no 2-7) | 0.482 | 0.24 | 1.00 | 0 | 0 |
| slow release (nf 7 tau 150) | 0.384 | 0.08 | 0.64 | 0.09 | 0.21 |
| slower release (tau 300-600) | 0.469 | 0.23-0.24 | 0.89-0.93 | 0-0.04 | 0.02-0.09 |
| weak drive (2-7 at 5) | 0.421 | 0.16 | 0.71 | 0.07 | 0.17 |

Without its release the reference scores 0.482, the same as the held double peaks the t-batch converges on
(0.48-0.50).

## Two things penalise a release that is not the reference's

- **Pace.**
  - With 7-2 at -12, every cycle recalls D, C, A, B in order, each alone and each released. That is correct serial
    order, but each peak comes about 100 later than in the reference.
  - In cycle 1, C peaks when A is due, A when B is due, and B at 600 after X.
  - Because recall is checked at the reference's own times, f4 credits the wrong points.
  - An event's recall is also read at its own fixed time, before it has peaked, so its trough later earns no f6.
- **Depth.** A weaker release returns a recalled event to about the median, not 2 below it, so its trough earns
  about 0.
- **Slow release.** With nf 7 at tau 150, D, C and A still come out in order but a slot late. B slips into the next
  cycle, so recall runs out of phase with X.

So a weak or slow release scores below no release at all: 0.38-0.42 against 0.48. Only a release close to the
reference's in speed and depth pays. The trough helper itself (`troughBelowFieldAtPosition`) measures correctly; it
is the fixed recall times, and the 2-below-median scale, that make the release terms judge pace and depth rather than
order.

## What a 10-run batch under this evaluator evolved

The t-batch ran 20-21 October 2026, built from `f7caf34`: population 1000, 200 generations, from scratch. Its
folders are `data/Serial Order Fitness Test/2026-10-20 20h23m23s` to `23h31m27s`, and each run's interpretation is in
its `config/README.md`.

| Run | Best | f3 | f4 | f6 | What the best does |
|---|---|---|---|---|---|
| t1, t3, t4, t8 | 0.494-0.501 | 1.00 | 0.33-0.36 | 0 | X raises D and C together; both held to the end |
| t5, t10 | 0.501-0.504 | 1.00 | 0.34 | 0 | D peaks alone at its point, C joins; both held |
| t2 | 0.482 | 0.83 | 0.35 | 0 | D peaks before X, C joins; both held |
| t9 | 0.482 | 0.90 | 0.37 | 0 | D peaks before X, C and A join over the cycles; all held |
| t7 | 0.478 | 0.99 | 0.23 | 0 | X raises D; held to the end |
| t6 | 0.522 | 0.82 | 0.49 | 0 | D, then C, then A peak in turn, a slot late; released items go back to pre-shaped, never below the median |

- No best earns f4 by a loophole.
- All ten store the sequence as ordered pre-shapes and recall the first item, or the first two, on X.
- None releases a recalled peak into a trough (f6 0 in all ten).
- Their fitness, 0.48-0.52, is the reference's with its release removed (0.482).
- Only t6 steps from item to item, and its release, too slow and too shallow, earns nothing under the fixed recall
  times.
