# Serial order: a fitness that follows the reference design stage by stage

`SerialOrderFitnessTest` (`include/solutions/serial_order_fitness_test.h`, 20 October 2026) scores the serial-order
task by how the input and output fields respond at the end of every interval of a fixed schedule, after what a
reference design does there. The idea: the old serial-order fitness pays nothing until a whole mechanism exists, so
it should guide evolution better to pay for each stage of the behaviour. The apps run it as `--task serial-order-fitness-test`,
starting from serial order's template (input and output fields only). Its config is `config/solutions/serial-order-fitness-test.json`: serial order's, with the weights below.

## The reference and the schedule

The reference is the hand-built design of 18 October 2026, with the competition in the output field itself
(`tests/fixtures/serial-order-fitness-test-reference.dnf`; its dnf-composer replay is
`serial-order/architectures/handbuilt-inrange-replay.dnf` in the hand-off bundle). It holds `gk cg 2-7`, a connection
out of the output, so evolution can reach it only with connections out of the output allowed. The task's config turns
that on (`GenomeMutationConstants.allowOutputAsConnectionSource`, `output-field-cannot-be-a-connection-source.md`).

The schedule is the replay's: X (a boost of the input) at 0, D 180, C 120, A 20 and B 80 at 100, 600, 800 and 1800,
each for 50; then X at 2550, 3550 and 4350, starting recall cycles of 1000, 800 and 1500.

`createPhenotypeEnvironment()` lays the schedule down as timed Gaussian stimuli on the input: `tgs X`, 1e6 wide and not
circular, so it lifts the whole field evenly as a boost does, then `tgs D`, `tgs C`, `tgs A` and `tgs B`. The
evaluation runs on those stimuli, stepping to each interval end, and every saved solution carries them. So a saved
champion opened in dnf-composer replays exactly what was scored, with no replay script. The stimuli come after the
fields in the element list, so a field feels each one a step late, in the evaluation and in the replay alike. A test
saves the reference, reloads it alone and checks that it recalls D, C, A and B at the scoring times. Moving the
evaluation onto these stimuli, from boosts and pulses added and removed by hand, changed no genome's score by more
than 0.03.

What the reference does, probed in dnf-composer at the end of each interval:
- the input peaks at each event (height 7.6, width 6) and is back at -10 by the next interval end;
- the output stays below threshold through the demonstration, pre-shaped at every stored event, the earliest
  highest, about 6 above the rest of the field (at 1850: 180 -2.0, 120 -4.4, 20 -5.4, 80 -7.2, field -13.0);
- after each X its peaks rise and fall, in time after X:

| Cycle | D | C | A | B |
|---|---|---|---|---|
| 1 | 20-232 | 238-350 | 356-456 | 466-554 |
| 2 | 32-214 | 218-330 | 336-436 | 446-534 |
| 3 | 32-216 | 220-332 | 338-436 | 446-536 |

- each recalled event is pushed at least 7.6 below the rest of the field. Below the resting level, though, the
  troughs are only 2.2-10.8 deep, since after recall the latch lifts the field to about -4;
- the troughs' order (the earliest deepest) does not hold: at 400 after X it fails in cycles 2 and 3, at 500 it
  ties, and at 600 the 120 trough is deepest in every cycle.

## The terms

| Term | Scored | Helpers |
|---|---|---|
| f1 | input: a peak at each event, at rest at every other interval end | `oneBumpAtPositionWithAmplitudeAndWidth`, `closenessOfMeanActivationToRestingLevel` |
| f2 | output below threshold at every demonstration interval end | `noBumps` |
| f3 | output pre-shaped at the stored events, in order | `preShapeAboveFieldAtPosition`, `activationsDescendAcrossPositions` |
| f4 | the events recalled one at a time, in order, sampled through 600 after X; each peak paid only near its event and by its height | `oneBumpAtPositionWithAmplitudeAndWidth` times `closenessOfNearestBumpToPosition` times `heightCreditOfNearestBumpToPosition` |
| f5 | at each sample, the events not yet recalled pre-shaped, in order | as f3 |
| f6 | each recalled event released once its peak ends: half below threshold, all of it 2 below the field | `activationAtPosition`, `troughBelowFieldAtPosition`, times the event's recall |
| f7 | at 600 after X and at the cycle's end: no peak, every event below the field as far as it was recalled | `noBumps`, `troughBelowFieldAtPosition` times recall |

## Weights

The weights were equal until 20 October 2026. From-scratch run 1 (`data/Serial Order Fitness Test/2026-10-20
14h28m28s`) then showed what equal weights reward. A genome with no connections already earns 0.36 from f1, f2 and
f7. Evolution learned every shape term but recall, and the shapes-only plateau (0.77) sat only 0.02 below its best
(0.79), which recalled one item per cycle.

The terms are now weighted by how much of the task each carries:

| f1 | f2 | f3 | f4 | f5 | f6 | f7 |
|---|---|---|---|---|---|---|
| 0.05 | 0.05 | 0.10 | 0.35 | 0.20 | 0.20 | 0.05 |

- f4, the recalled peaks, is the task itself.
- f5 and f6 are what turn a peak held through a cycle into stepping through the sequence: the events still to come
  stay ordered, and those already recalled are suppressed.
- f3 is the memory everything starts from. Evolution found it within about 10 generations.
- f1, f2 and f7 are mostly constraints that genomes satisfy with little structure.

Scores on the same genomes:

| Genome | Equal weights | These weights |
|---|---|---|
| reference | 0.989 | 0.972-0.976 |
| run 1's best (one peak per cycle) | 0.792 | 0.677 |
| X lifts D each cycle and holds it (made-up partials) | 0.713 | 0.624 |
| run 1's shapes-only plateau (gen 100) | 0.769 | 0.578 |
| `serial-order-single-recall.dnf` | 0.750 | 0.519 |
| run 1's static troughs (gen 3) | 0.520 | 0.332 |
| burst (`sequence-timing-boost-burst.dnf`) | 0.360 | 0.275 |
| no connections | 0.357 | 0.125 |

The stop rule is unaffected: it compares each term with the target, not the weighted sum.

## Recall peaks are paid only near the event due

- Under these weights, the first two runs of the weighted batch (w1 and w2, 20 October 2026) converged on one output
  peak. It rose before any X and was never let go: w1's at 138, near no event, and w2's at C.
- The bump score gives 0.45 for one bump and 0.10 for its height and width, wherever the bump is. So that peak earned
  f4 0.53-0.63 and beat run 1's real recall. The batch was stopped after those two runs.
- f4 now multiplies each recall point's bump score by `closenessOfNearestBumpToPosition`. That is 1 when the output's
  nearest bump lies within 2 of the event due, and exp(-(d - 2)^2 / 8) beyond.
- The reference keeps f4 0.92. The two held peaks are test fixtures (`serial-order-fitness-test-held-peak-*.dnf`).

Under these weights, old rule against new:

| Genome | f4 old | f4 new | Fitness old | Fitness new |
|---|---|---|---|---|
| reference | 0.93 | 0.92 | 0.976 | 0.972 |
| made-up: D recalled, then suppressed, each cycle | | 0.23 | | 0.626 |
| run 1's best (one peak per cycle) | 0.39 | 0.15 | 0.677 | 0.594 |
| run 1's shapes-only plateau (gen 100) | 0.00 | 0.00 | 0.578 | 0.578 |
| w2's best (peak held at C) | 0.63 | 0.24 | 0.701 | 0.564 |
| made-up: D lifted and held each cycle | | 0.23 | | 0.498 |
| w1's best (peak held at 138) | 0.53 | 0.00 | 0.678 | 0.493 |
| burst (`sequence-timing-boost-burst.dnf`) | 0.48 | 0.00 | 0.275 | 0.107 |

- A held peak now ranks below the shapes-only plateau.
- Recalling the first item and then suppressing it ranks above the plateau, but only by about 0.05.
- Part of the plateau's value is troughs that come without recall, the open point below.

## Recall peaks are paid by their height

Three of the first four runs of the gated batch found the next loophole (g1, g2 and g3, 20 October 2026). Their output
sits right at threshold and crosses it by 0.0-0.5 at A or B when they are due. Height was a twentieth of the bump
score, so each crossing earned about 0.9 of the point: f4 0.23-0.38 with no real peak. The batch was stopped after g4.

Each recall point is now also multiplied by `heightCreditOfNearestBumpToPosition`: the bump's height over 2, at most 1.
This is the same rule serial order's evaluator uses. The reference's recalled peaks are 7-12 high, so it is unchanged.
g2's best is the fixture `serial-order-fitness-test-threshold-flicker.dnf`.

| Genome | f4 now | Fitness before | Fitness now |
|---|---|---|---|
| reference | 0.92 | 0.972 | 0.972 |
| made-up: D recalled, then suppressed, each cycle | 0.23 | 0.626 | 0.626 |
| run 1's best (one peak per cycle) | 0.15 | 0.594 | 0.594 |
| g4's best (peaks 12-13 high, C then A, A held) | 0.31 | 0.590 | 0.590 |
| run 1's shapes-only plateau | 0.00 | 0.578 | 0.578 |
| w2's best (peak held at C) | 0.24 | 0.564 | 0.564 |
| g1-g3's bests (flicker at threshold) | 0.01-0.03 | 0.587-0.638 | 0.509-0.526 |
| w1's best (peak held at 138) | 0.00 | 0.493 | 0.446 |

Every loophole found so far now ranks below the shapes-only plateau, and the two designs with real recall rank above
it.

## Troughs are paid only at events recalled first

The height-scaled batch's first four runs (h1-h4, 20 October 2026) all leaned on troughs at events that never peaked:
- h1 and h2 dropped D and C before any X, then held a real peak at A;
- h3 and h4 sank D and C into troughs during recall without recalling anything.

f6 paid troughs at every earlier event whether or not it had been recalled, which was worth f6 0.89, about 0.18 of the
total. Dropping D and C early also took them out of recall altogether. The batch was stopped after h4.

Now, at each recall point, the evaluator records how fully the event due is recalled. That is the closeness of the
nearest bump times its height credit, 1 for a real peak at the event. f6 and the trough half of f7 weight each
event's trough by its own recall earlier in that cycle. The reference keeps f6 and f7 at 1.0. h3's best is the
fixture `serial-order-fitness-test-troughs-without-recall.dnf`, pinned at f6 0 and f7 0.5.

| Genome | f4 | f6 | f7 | Fitness before | Fitness now |
|---|---|---|---|---|---|
| reference | 0.92 | 1.00 | 1.00 | 0.972 | 0.972 |
| made-up: D recalled, then suppressed, each cycle | 0.23 | 0.61 | 0.63 | 0.626 | 0.626 |
| w2's best (peak held at C) | 0.24 | 0.00 | 0.00 | 0.564 | 0.436 |
| run 1's best (one peak per cycle, not suppressed within the cycle) | 0.15 | 0.00 | 0.48 | 0.594 | 0.424 |
| g4's best (C then A, A held) | 0.31 | 0.09 | 0.06 | 0.590 | 0.420 |
| h1-h2's bests (D and C dropped, A held) | 0.22-0.23 | 0.00 | 0.00 | 0.571-0.590 | 0.380-0.400 |
| shapes only (run 1 gen 100, h3, h4) | 0.00 | 0.00 | 0.50 | 0.565-0.578 | 0.375-0.386 |
| g2's best (flicker) | 0.01 | 0.00 | 0.40 | 0.509 | 0.319 |

Everything found so far now falls between 0.32 and 0.44. The first step of the real mechanism, recalling an item and
then suppressing it, is worth about 0.19 more than the best of them.

Choices made with the user:
- the recall interval ends were moved inside the reference's peaks on every cycle. The first table put them at 200,
  350, 450 and 550 after X, where the reference is mid-switch or already on the next event;
- the boost intervals are not scored;
- pre-shapes and troughs are measured against the field's median, not its resting level. `preShapednessAtPosition`
  scores a level (closeness to h/2), so a field lifted as a whole scored as pre-shaped everywhere: at 150 the
  reference's stored D scored 0.88 and every unstored position 1.0. Two new helpers measure against the median, and
  the old ones are unchanged;
- a pre-shape row is its mean pre-shape times (0.5 + 0.5 x order), so noise ordering a flat field earns nothing;
- trough order is not scored.

## Recall is scored by order, not at fixed times

- The t-batch (10 runs on `f7caf34`) stored the order every time and recalled the first one or two items, but never
  released a recalled peak. All ten scored 0.48-0.52, the reference without its release.
- Weakening the reference's done loop showed why (`serial-order-release-evaluation.md`). Recall was checked at the
  reference's own times, 150, 290, 400 and 500 after X. A release that was slower or shallower than the reference's
  scored below no release at all: 0.38-0.42 against 0.48.
- The reference with its done field's inhibition weakened to -12 recalls in perfect order every cycle, but about 100
  slower. It scored 0.58.

Since then, each recall cycle is sampled every 10, from X to 600 after it. At each sample, each event's recall score is
the bump score times the closeness and height credit of the output's nearest bump. An event counts as peaking from
0.25.
- **f4:** each event's best recall score in the cycle, counted only if it first peaked in its place: D before any other,
  and each later event strictly after the one before it. Events already up when the cycle starts, held from the cycle
  before, all peak "at once", so only the first counts.
- **f5:** at each sample, the events that have not yet peaked must be pre-shaped in order. A cycle where no event ever
  waits scores 0.
- **f6:** once an event's first peak has ended, each later sample scores 0 if it is above threshold, half if below, and
  all of it 2 below the field. The cycle's f6 is the mean over the events of recall times mean release.
- **f7:** unchanged, at 600 after X and at the cycle's end.
- **The trace** keeps its 27 snapshots, and adds each cycle's first peaks, best scores, recall and release.

Fixtures:
- `serial-order-fitness-test-weak-release.dnf` (the reference with 7-2 at -12);
- `-late-stepping.dnf` (t6);
- `-held-double-peak.dnf` (t1).

Weakening the release now lowers fitness smoothly, from the reference's 0.969 to 0.482 with no release:

| Weakening | Fitness, intact first |
|---|---|
| 7-2: -30, -20, -16, -12, -10, -8 | 0.969, 0.966, 0.947, 0.849, 0.711, 0.484 |
| nf 7 tau: 75, 100, 150, 300 | 0.969, 0.864, 0.651, 0.556 |
| 2-7: 8, 6, 5, 3 | 0.969, 0.716, 0.513, 0.482 |

| Genome | f4 | f5 | f6 | f7 | Fitness before | Fitness now |
|---|---|---|---|---|---|---|
| reference | 0.96 | 0.96 | 0.96 | 1.00 | 0.972 | 0.969 |
| reference, weak release (7-2 -12) | 0.96 | 0.97 | 0.48 | 0.48 | 0.580 | 0.849 |
| t6 (steps D, C, A late, no suppression) | 0.56 | 0.96 | 0.19 | 0.00 | 0.522 | 0.593 |
| t5, t9, t1, t7 (held peaks) | 0.23-0.32 | 0.91-0.99 | 0.00 | 0.00 | 0.478-0.504 | 0.461-0.484 |
| h3 (troughs, no recall) | 0.00 | 0.42 | 0.00 | 0.50 | 0.384 | 0.302 |
| run 1's best (C, then A: out of order) | 0.00 | 0.34 | 0.05 | 0.48 | 0.424 | 0.295 |
| w2 (C held from before X) | 0.00 | 0.50 | 0.00 | 0.00 | 0.436 | 0.274 |
| g4 (C then A, A held) | 0.08 | 0.23 | 0.08 | 0.06 | 0.420 | 0.248 |
| h1 (D, C dropped; A held) | 0.00 | 0.19 | 0.00 | 0.00 | 0.400 | 0.224 |
| g2 (flicker) | 0.00 | 0.22 | 0.00 | 0.40 | 0.319 | 0.193 |

- Stepping through the items, even late and without suppression, now ranks above holding them.
- Recall out of order now earns nothing.

## The input only detects, memory is held below threshold, and recall is one peak at a time

The o-batch (`8f85eb7`, 21 October 2026) showed three ways to earn credit that the task does not intend:
- **Input as memory.** o1, o2 and o5 hold the four items as self-sustained peaks in the input field. f1 weighed 0.05 and
  looked at the input only before each next onset and at four recall points, so this cost about 0.025. A peak that
  lingers after its pulse but dies before the next went unseen.
- **Output peaks while learning.** t6, o4 and o1 peak in the output during the demonstration. f2 also weighs 0.05 and
  was checked at 9 points; f3 paid a held peak as memory.
- **Bursts.** o2, o3 and o6 push all four items over threshold within 20-50 of X, highest pre-shape first, and let them
  fade together. Sampled every 10, they peak "in order", and each earned 0.6-0.9 of a recall point, at a small discount
  for several bumps. o3's 0.737 beat o1's real stepping, 0.670.

Since then:
- Every stage is sampled every 10. The input must be at rest at every sample outside a pulse and the 50 after it (five
  time constants of the reference's input).
- **f3** counts each sample's pre-shapes only as far as the input is at rest and the output has no peak.
- **f5** counts each sample's waiting pre-shapes only as far as the input is at rest. Inside a pulse and the 50 after
  it, the input's last score before the pulse stands in, since X lifts the whole input.
- **f4** shares each sample's recall score among the output's peaks at that moment: alone, it counts in full; beside one
  other, half; in a burst of four, a quarter. The user chose graded over all-or-nothing scoring here: evolution needs a
  slope to climb.

Fixtures: `serial-order-fitness-test-input-holds-items.dnf` (o1) and `-burst.dnf` (o2).

Scores, 2 evaluations each, before (`8f85eb7`) and after:

| Genome | Before | After | What changed |
|---|---|---|---|
| reference | 0.969 | 0.969 | |
| weak release | 0.849 | 0.849 | |
| o3 (burst, input at rest) | 0.737 | 0.662 | f4 0.69 to 0.45 |
| t6 (steps late, peak at D while learning) | 0.592 | 0.518 | f3 0.83 to 0.60, f4 0.56 to 0.47 |
| o5 (steps one at a time, input holds items) | 0.702 | 0.503 | f2 1.00 to 0.73, f3 0.70 to 0.20, f5 0.95 to 0.27 |
| o4 (burst beside a held D) | 0.582 | 0.478 | f3 0.95 to 0.68, f4 0.55 to 0.35 |
| o1 (steps one at a time, input holds items) | 0.670 | 0.471 | f3 0.59 to 0.08, f5 0.93 to 0.26 |
| t1-t10 (first two items peak together) | 0.46-0.49 | 0.38-0.46 | f4 0.18-0.32 to 0.09-0.23 |
| o2 (burst, input holds items) | 0.666 | 0.335 | f3 0.98 to 0.11, f4 0.54 to 0.32, f5 0.94 to 0.11 |
| h4, g1, g2 (input holds items) | 0.19-0.29 | 0.10-0.17 | |

- The four bests the user judged closest to solving (o1, o4, o5, t6) now rank 2nd to 5th among the evolved genomes,
  above every t.
- o3 still ranks first. Its input and memory are clean (f1 0.96, f2 1.00, f3 1.00, f5 0.91), and its common fade still
  earns f6 0.49, because release is weighted by each event's peak height, not by the recall credit it shared.

## No all-or-nothing scores

The user's rule (21 October 2026): graded scoring is always better, because evolution has something to move towards.
Four cut-offs were replaced:
- **Order.** An event out of its place earned nothing. Now an event's credit is its best recall score times its place
  (`tools::utils::shareInPlace`):
  - for D, the share of the other events that did not peak before it;
  - for any later event, the share of the events due before it that peaked before it.
  A cycle entered at A (A, B, D, C) gives D 1/3, C 1, A 0 and B 1/3.
- **Release.** A released event scored 0 above threshold and jumped to 1/2 just below it. Its below-threshold half now
  ramps from none 2 above threshold to all of it 2 below (`Solution::belowThresholdAtPosition`); the other half is
  still its depth below the field.
- **Release and troughs are weighted by the event's credit in place**, not by its peak's height. A burst's common fade
  now earns only as much as the burst earned recall.
- **f5** skips a cycle where every event peaked by the first sample, rather than scoring it 0.

The 0.25 recall score that marks an event's first peak stays: it only times the peak, and the credit is graded.

| Genome | `8f85eb7` | `c9e40cc` | Now |
|---|---|---|---|
| reference | 0.969 | 0.969 | 0.960 (f6 0.93, f7 0.98) |
| weak release | 0.849 | 0.849 | 0.845 |
| o3 (burst, input at rest) | 0.737 | 0.662 | 0.609 (f6 0.49 to 0.22) |
| t6 | 0.592 | 0.518 | 0.501 |
| o5 | 0.702 | 0.503 | 0.469 |
| o1 | 0.670 | 0.471 | 0.463 |
| o4 | 0.582 | 0.478 | 0.451 (f6 0.17 to 0.04: D never lets go) |
| t1-t10 | 0.46-0.49 | 0.38-0.46 | 0.38-0.46 |
| o2 | 0.666 | 0.335 | 0.268 |

## The stop weighs more, and silence counts only after recall (`a892c59`)

The release loop sat behind a valley. From an adaptation loop, adding the done field's memory without its reset lost
fitness (0.549 to 0.474). Only f7, the stop, rewards that memory, and f7 weighed 0.05.
- Weights [.05 .05 .10 .35 .20 .20 .05] became [.05 .05 .10 .30 .10 .20 .20].
- f7's half for an output with no peak now counts only as far as the cycle recalled in order. A silent output, which
  earned 0.5, earns nothing.

Re-scored on the reference's variants:
- no loop 0.370;
- adaptation loop 0.431;
- adding the memory without the reset 0.456;
- whole loop 0.960.

`serial-order-search-and-hyperparameters.md` has the analysis.

## Scores, 5 evaluations each, under equal weights and the old f4

| Genome | Fitness | Notes |
|---|---|---|
| reference | 0.988 (0.986-0.988) | every term 1.0 but f4 0.92: some peaks centre half a sample off, and position credit is 1 / (1 + distance) |
| `serial-order-single-recall.dnf` | 0.75 | f4 0.12: the done field is never cleared, so only the first cycle recalls |
| `serial-order-overlapping-peaks.dnf` | 0.63 | |
| `serial-order-low-peaks.dnf` (output-feedback run 3) | 0.57 | |
| `serial-order-chaotic-sweep.dnf` | 0.43 | |
| `serial-order-handbuilt.dnf` (the reachable 8-field design) | 0.41 | f3 0: its output copies the selection field, so it never pre-shapes |
| mc6, the best from-scratch champion | 0.41 | |
| sc5, the 7-field latch-as-done design | 0.40 | |
| `serial-order-threshold-flicker.dnf` | 0.38 | |
| bursts and fixed lag (mc1, `sequence-timing-boost-burst.dnf`, `-fixed-lag.dnf`) | 0.32-0.36 | f4 still 0.33-0.59 |

- An output with no input connections scores f2 1.0 and f7 about 0.5, and nothing on f3-f6.
- The fixed schedule makes evaluations nearly noise-free (spread 0.002 on the reference).
- Open:
  - one fixed gap and no probe trial without X leave room for fixed-lag replay.
  - (Fixed, see above.) Troughs were paid for without recall. In a 10-generation smoke run (population 200, 20 October 2026), the best
    (0.456) had a slow output (tau about 6000) and a hidden field that stores D and inhibits the output. So a trough
    at D deepens through the whole trial, and earns f6 0.61 and f7 0.63 with no peak ever rising.
- A genome with only the input and output, and no connection, already scores about 0.35: f1 about 0.96, f2 1 and
  f7 0.5.
