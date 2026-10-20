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
| f4 | the right recalled peak at 150, 290, 400 and 500 after X, paid only near the event due | `oneBumpAtPositionWithAmplitudeAndWidth` times `closenessOfNearestBumpToPosition` |
| f5 | events still to recall pre-shaped, in order | as f3 |
| f6 | recalled events pushed below the field | `troughBelowFieldAtPosition` |
| f7 | at 600 after X and at the cycle's end: no peak, every event below the field | `noBumps`, `troughBelowFieldAtPosition` |

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
  - 90% of a peak's credit is count and position, so a peak barely above threshold earns almost full credit;
  - one fixed gap and no probe trial without X leave room for fixed-lag replay.
  - Troughs are paid for without recall. In a 10-generation smoke run (population 200, 20 October 2026), the best
    (0.456) had a slow output (tau about 6000) and a hidden field that stores D and inhibits the output. So a trough
    at D deepens through the whole trial, and earns f6 0.61 and f7 0.63 with no peak ever rising.
- A genome with only the input and output, and no connection, already scores about 0.35: f1 about 0.96, f2 1 and
  f7 0.5.
