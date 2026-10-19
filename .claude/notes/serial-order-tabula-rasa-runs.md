# Serial order: tabula rasa runs (18-19 October 2026)

Experiment report for evolving the `serial-order` task from input and output fields only. Task and hand-built
reference: `serial-order-handbuilt-design.md`. Earlier from-scratch failures on the timing task:
`sequence-timing-why-tabula-rasa-fails.md`.

These runs used the first serial-order demonstration (D, C, A, B at 100, 600, 800, 1800 after X) and the old
isolation check. On 19 October 2026 the demonstration became evenly spaced (`cbbf9f6`), and each peak now has to rise
alone (`51d03d9`). Their scores are not comparable with runs from then on.

## Summary

- **No run solved the task.** Eleven runs across two batches, none above 0.71 when re-evaluated. The hand-built
  design scores 0.993.
- **First batch (3 runs):** reached partial order, but through output active all over the field. One run found a new
  loophole, the chaotic sweep, which the evaluator now closes.
- **Second batch (8 runs, with the activity gate):** every run settled on a burst, with all four events firing together
  just after X, at 0.58-0.62. None got past 0.12 order credit. Stopped by request after run 8 had started.
- **No run built either mechanism the task needs:** a store whose recall is ordered, and suppression of each item
  once recalled.

## Setup

- Population 1000, 200 generations, one run at a time, `--target 0.93`. The stop rule needs every partial term above
  the target, and the hand-built design's f1 is 0.939, so 0.98 could never fire.
- `config/solutions/serial-order.json` is the boost-order config of sequence-timing runs 24-33 with two ranges widened
  so that every setting of the hand-built design is reachable (`7d3ae9b`): resting level up to +1.0, and field
  Gauss-kernel global down to -1.0. A test guards this.
- One serial-order evaluation takes about 205 ms for a 7-field genome. A run of 200 generations took 1 h 54 min to
  2 h 53 min.

## First batch: before the activity gate

| Run | Reported best | Re-evaluated | Behaviour |
|---|---|---|---|
| 1 | 0.715 | 0.706 | Cycles 2-3: D, then C about 410 later, then A; B fires with D (order 0.67). Cycle 1 a burst |
| 2 | 0.807 | 0.67 | Chaotic sweep (below) |
| 3 | 0.687 (stopped at gen 86) | - | Burst: the whole output field just above threshold after X |

**The chaotic sweep (run 2).** For about 450 after each X the output holds 10-21 broad peaks (widths up to 100,
activation up to 120) all over the field, which cross the event positions by chance. The evaluation trace counted
47-93 rising peaks per cycle. Selecting the best of 1000 single noisy evaluations kept its lucky draws, so its
reported best climbed while its true fitness stayed near the burst. Fixture: `tests/fixtures/serial-order-chaotic-sweep.dnf`.

**The fix (`7ea01c6`, serial-order only):** each recall cycle's f4-f6 are scaled by
`SequenceTiming::activityAtEvents`. It takes the share of active output sites (summed over every step) within 10 of an
event, and corrects it for chance: (share - chance) / (1 - chance), floored at 0. Chance is 0.42, because the four
windows cover 84 of 200 sites. A first version counted peak centroids instead of sites; it let a broad peak whose
centre happened to lie near an event pass, and scored the sweep 0.3-0.5 per cycle. After the fix:

| Solution | Before | After |
|---|---|---|
| Hand-built design | 0.993 | 0.993 |
| Run 2, chaotic sweep | 0.67 | 0.33 |
| Run 1, grouped order | 0.706 | 0.47 (cycles 2-3 carry a broad slab of activity over positions 20-160) |
| Run 3, whole-field burst | 0.687 | 0.35 |
| Output that never peaks | - | about 0.33 |

## Second batch: with the activity gate

| Run | Reported best | Re-evaluated | Order credit | Behaviour |
|---|---|---|---|---|
| 1 | 0.599 | 0.56 | 0 | All four together, faint peaks (activation 0.2-2.3 in dnf-composer) |
| 2 | 0.622 | 0.615 | 0.07 | Two simultaneous pairs: A with D, then B with C |
| 3 | 0.673 | 0.593 | 0.06 | Burst with onsets that vary by cycle; the reported best was inflated by noise |
| 4 | 0.598 | 0.50 | 0 | All four together; some cycles miss events |
| 5 | 0.619 | 0.603 | 0.12 | D first, then A, B and C together about 30 later |
| 6 | 0.623 | 0.582 | 0.03 | All four together |
| 7 | 0.609 | - | 0.05 | All four together |
| 8 | 0.506 (stopped at gen 30) | - | 0 | A and B together |

- Every champion is active only at the event positions (activity score 0.99-1.0), and silent without X.
- f6 stays at about 0.4: the last peak dies out, and isolation is 0.2-0.3 with four peaks up at once.
- The burst earns f4 = 1 with peaks barely above threshold. **Peak height is not scored** - an open evaluator gap.
- Most runs plateaued within 50-80 generations. Species counts swung between 1 and several hundred, with repeated
  collapses, as in sequence-timing runs 24-33.

## What this says

- Serial order alone is not easier for from-scratch evolution than order plus timing. The burst is the same local
  optimum, at a similar height (about 0.6).
- With off-event activity no longer paid, there is no partial credit on the way to ordered recall. A burst earns full
  f4, and any step towards a primacy gradient or a done field breaks the burst before it orders anything.
- Re-evaluate every reported best before trusting it: noisy evaluation plus best-of-1000 selection inflated reported
  bests by up to 0.14.

## Where the data is

`neat-dnfs/data/Serial Order/`:
- First batch: folders `2026-10-18 09h42m10s` to `16h23m12s` (the last is an accidental run, stopped at generation 3).
- Second batch: `2026-10-18 16h32m43s` to `2026-10-19 09h14m10s`.

Each holds `config/README.md` with the outcome. Champion copies and dnf-composer replays are under
`.claude/temp/so1-so3` and `sa1-sa6` (gitignored scratch). Queue scripts: `.claude/temp/queue14.sh` and `queue15.sh`.
