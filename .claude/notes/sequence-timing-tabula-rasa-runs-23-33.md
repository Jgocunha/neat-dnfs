# Sequence timing: tabula rasa runs 23-33 (17-18 October 2026)

Experiment report. The assessment it led to (why from-scratch evolution fails and what to change) is in
`sequence-timing-why-tabula-rasa-fails.md`.

## Summary

- **No run solved the task.** Solved means a best fitness of 0.98; the hand-built reference scores 0.988. The best runs
  reached 0.824 (run 23, through a loophole) and 0.740 (run 29, without it).
- **Run 23 found a loophole: fixed-lag replay.** It replays the demonstrated order at a fixed delay after the
  demonstration, not after X, and also fires when X never comes. The probe gate (`0cf8fb5`) closed it.
- **With the gate (runs 24-33), every best solution needs X.** Six of ten stayed in the burst, with all events firing
  together right after X. Four reached partial order by firing the events in groups.
- **No run built the timing mechanism.** No run gave each event its own delay after X, and no run switched recall
  peaks off.

## Setup

**Task:** `sequence-timing-boost-order`. The demonstration delivers the start cue X as a boost of the whole input field,
then the events D, C, A and B as Gaussian pulses at 100, 600, 800 and 1800 after X. After a gap, X alone is given again.
The output field must then peak at D, C, A and B in turn, at the demonstrated delays after X, one peak at a time. Each
evaluation runs three recall trials, with gaps drawn afresh from each third of 300-1500. It also runs one probe trial
in which X never comes.

**Fitness: seven weighted terms.**

| Term | Measures | Weight |
|---|---|---|
| f1 | Input field forms a bump at each pulse | 0.095 |
| f2 | Input field returns to rest between events | 0.095 |
| f3 | Output silent outside recall | 0.095 |
| f4 | Recalled events: a peak rises at each event after X | 0.19 |
| f5 | Timing: half order credit, half exact onsets after X | 0.285 |
| f6 | One peak at a time, and the last peak dies out | 0.19 |
| f7 | Output stays near its resting level | 0.05 |

Order credit gives each consecutive pair credit for rising after its predecessor, full credit at 100 or more apart.
0.67 therefore means two of the three pairs.

**Evaluator changes during the experiment:**
- `d16e95f`, before run 23, added f7. Its credit is full while the output's lowest activation averages within 60
  below rest. It came from run 20, whose output sat about 650 below rest.
- `0cf8fb5`, between runs 23 and 24, added the probe gate: f4-f6 are multiplied by how silent the output stays in the
  probe trial without X.

**Hyperparameters:** the committed `config/solutions/sequence-timing-boost-order.json`, unchanged in every run:
- compatibility threshold 1.5
- species culled after 15 generations without improvement
- new-field probability 0.01 and new-connection probability 0.3 per genome per generation
- tau drawn log-uniformly from 5 to 1e5, mutated by x1.3
- Gauss connection amplitude from 1 to 1000, mutated by x1.3
- resting level from -15 to -0.5
- population 1000, starting from input and output fields only

Run 23 ran 200 generations. Runs 24-33 ran 100 generations each, one after another, each set to stop early at 0.98
(`neat-dnfs-evol --task sequence-timing-boost-order --runs 1 --pop 1000 --gens 100 --target 0.98`). Run 33 was stopped
by request at generation 82. Each run took about 1.9 hours per 100 generations on its own.

## Run 23: fixed-lag replay (before the probe gate)

- **Progress.** The best rose from 0.42 to 0.65 by generation 30, jumped to 0.77 at generations 57-58, then sat near
  0.81 until the end. Its peak was 0.824 at generation 188.
- **Timing came from the demonstration.** D, C and A rose at the same absolute time whatever the gap. In dnf-composer
  replays with X at 2150 and at 3350, D rose at t = 3250 in both. With a gap of 1500, D rose before X.
- **Order was right, intervals were not.** At gaps near 1100, the fixed lag happened to put the onsets close to their
  targets: 118, 588, 746, 1988 after X, against 100, 600, 800, 1800. At a gap of 323 they came at 826, 1256, 1380, 1574.
- **Peaks.** They were slow ramps barely above threshold, growing from amplitude 0.02 to about 3 over 2000 time units,
  and none switched off.
- **It also fired without X.** The output was only 52% silent in the probe trial, but the probe only counted inside
  f3, which still scored 0.90.
- **Under the probe gate**, its champion scores about 0.56. It is now the test fixture
  `tests/fixtures/sequence-timing-boost-fixed-lag.dnf`.

## Runs 24-33: with the probe gate

| Run | Best | f5 | f6 | Order credit per trial | Behaviour | Onsets after X (D, C, A, B), first trial |
|---|---|---|---|---|---|---|
| 29 | 0.740 | 0.41 | 0.53 | 0.67, 0.33, 0.42 | grouped | 708, 850, 708, 850 (groups change with the gap) |
| 31 | 0.731 | 0.45 | 0.43 | 0.67, 0.67, 0.67 | grouped | 30, 1710, 1810, 1710 |
| 25 | 0.722 | 0.44 | 0.43 | 0.67, 0.67, 0.67 | grouped | 1592, 1702, 1592, 1702 |
| 33 | 0.696 | 0.44 | 0.27 | 0.67, 0.67, 0.53 | grouped | 882, 1088, 882, 1088 (stopped at generation 82) |
| 30 | 0.648 | 0.22 | 0.39 | 0.38, 0.39, 0.38 | burst | 48, 166, 44, 58 |
| 32 | 0.640 | 0.24 | 0.37 | 0.53, 0.26, 0.29 | burst | 36, 120, 36, 110 |
| 26 | 0.628 | 0.24 | 0.20 | 0.26, 0.25, 0.21 | burst | 100, 36, 36, 114 |
| 24 | 0.626 | 0.13 | 0.39 | 0, 0, 0.11 | burst | 94, 54, 54, 52 |
| 27 | 0.623 | 0.13 | 0.45 | 0.07, 0.18, 0.12 | burst | 68, 68, 68, 90 |
| 28 | 0.601 | 0.11 | 0.38 | 0, 0.01, 0.01 | burst | 90, 46, 46, 44 |

Target onsets after X: D 100, C 600, A 800, B 1800. Each row shows the best solution of its run.

- **The gate worked.** All ten best solutions stay fully silent in the probe trial without X.
- **Burst (6 of 10).** All four events ignite 30-170 after X. This earns full recalled-events credit and some order
  credit by chance, and scores 0.60-0.65.
- **Grouped order (4 of 10).** The events fire in two or three groups, so some consecutive pairs earn order credit.
  - Run 31 is the closest to the target structure. D fires 30 after X. C and B fire together 1650-1710 after X, and A
    about 100 later. The pattern is the same across all three gaps, so it is timed from X.
  - Runs 25 and 33 fire D and A together, then C and B together. Their onsets move with the gap, so part of their
    timing still runs from the demonstration.
- **No run switched peaks off.** f6 stays between 0.20 and 0.53. About 0.42 is the score for peaks that pile up
  (isolation 1, 1/2, 1/3, 1/4, and the last peak never dies).
- **Structure did grow.** Populations ended with 6-8.5 fields per genome on average, against 2 at the start. Runs ended
  with 166-665 species, except run 27, which collapsed to a single species (as did run 23).
- **Most gains came early.** The grouped runs first passed 0.70 at generations 53-83 (run 33 never did) and then
  flattened.

## Earlier from-scratch runs, for context

These used older evaluators (no probe gate, no f7), and runs 12-15 still had the elitism bug, so scores are only roughly
comparable.

| Run | Setup | Best | Outcome |
|---|---|---|---|
| 12 | boost, threshold 4.0 | 0.696 | Burst; one species and 2 fields throughout |
| 14 | boost, threshold 1.5 | 0.654 | Grew 6 fields; burst persisted |
| 15 | boost-order, threshold 1.5 | 0.650 | One ordered pair after generation 130, which depended on the gap |
| 16 | order credit only | 0.670 | Escaped the burst, then plateaued |
| 19 | full task, seeded from run 16 | 0.629 | The seed gave no head start |

## Where the data is

- Runs: `neat-dnfs/data/Sequence Timing Boost Order/`, folders `2026-10-17 07h52m18s` (run 23) to
  `2026-10-18 07h44m41s` (run 33). Each holds `config/README.md` with the outcome and a copy of the config used.
- Run 23 replays for dnf-composer: `.claude/temp/analysis/sequence-timing-boost-run23-evolved-replay-gap300.dnf` and
  `-gap1500.dnf` (gitignored scratch).
- Per-run best, partial terms and onsets come from each run's `overview.jsonl`: `bestSolution.partialFitness` and
  `bestSolution.evaluationTrace.trials[].recall.onsets`.
