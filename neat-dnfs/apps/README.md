# Examples

Three executables, each built from one `.cpp` file here plus the shared
[`solution_registry.h/.cpp`](solution_registry.h) (task lookup + CLI parsing).
Binaries are found in your build output directory (e.g. `build/Debug/` on
Windows, `build/` on Linux/macOS).

| Executable | Source | What it does |
|---|---|---|
| `neat-dnfs-evol` | `neat_dnfs_evol.cpp` | Evolves a task from a minimal (or ablation-seeded) genome |
| `neat-dnfs-inc-evol` | `neat_dnfs_incremental_evolution.cpp` | Evolves a task starting from a loaded template solution |
| `neat-dnfs-sol-eval` | `neat_dnfs_solution_evaluation.cpp` | Loads a template solution and repeatedly calls `evaluate()` — no evolution |

Every binary supports `--list` (print available tasks and ablations) and
`--help` (print all flags).

## Flags

| Flag | Applies to | Meaning | Default |
|---|---|---|---|
| `--task NAME` | all | Task to run (see `--list`) | `selection-instability` (evol) / `ior` (inc-evol, sol-eval) |
| `--ablation NAME` | evol, inc-evol | Ablation preset to apply before `initialize()` (see `--list`) | none (baseline) |
| `--template PATH` | inc-evol, sol-eval | Template JSON to load a starting solution from. Each connection evolution could not have made (one out of an output field, say) is logged as a warning | the task's own template under `templates/` |
| `--runs N` | evol, inc-evol | Number of independent `Population::evolve()` runs | `SolutionConstants.numberRuns` from config |
| `--pop N` | evol, inc-evol | Population size per run | `SolutionConstants.populationSize` from config |
| `--gens N` | evol, inc-evol | Max generations per run | `SolutionConstants.numberGenerations` from config |
| `--target F` | evol, inc-evol | Ends a run early once every partial fitness of the best solution exceeds F (a task whose fitness is the larger of two scores checks the terms of the larger) | `SolutionConstants.targetFitness` from config |
| `--evals N` | sol-eval | Number of `evaluate()` calls | 20 |
| `--config PATH` | all | Reference hyperparameter JSON | `config/neat_dnfs.json` |

Available tasks: `and`, `xor`, `detection-instability`, `memory-instability`,
`selection-instability`, `memory-trace`, `dmts`, `ior`, `hri-packaging`, `sequence-timing-behaviour`,
`sequence-timing-behaviour-guided`, `serial-order-behaviour`, `serial-order-behaviour-guided`,
`sequence-and-timing-recall`, `incremental-recall-1-single-order`, `incremental-recall-2-single-order-and-timing`,
`incremental-recall-3-repeatable-order`, `incremental-recall-4-repeatable-order-and-timing`,
`incremental-recall-5-consecutive-order`.

Available ablations: `no-growth-io-only`, `no-growth-reference-hidden-field-count`,
`no-speciation`, `no-crossover`, `random-initial-topology`, `compat-threshold-4`, `output-feedback`,
`timing-search-levers`.

The two `serial-order-*` tasks run the same trial and differ only in how they score it, each from
its own `config/solutions/<task>.json`:
- `serial-order-behaviour` scores what the output does: it stays quiet while the sequence is shown,
  recalls D, C, A and B in order on each cue, releases each recalled event below threshold, and stops;
- `serial-order-behaviour-guided` also scores what the reference design holds below threshold: the
  events pre-shaped in order, and each recalled event pushed below the field.

The two `sequence-timing-behaviour*` tasks score the same way, and also when each event is recalled: after X, D, C, A
and B must peak at the times they were shown, 100, 600, 800 and 1800 after it. Each evaluation runs a probe trial
without X, as long as its longest recall trial, then three recall trials after gaps drawn afresh from 300-1500, each
from rest. Any peak at an event in the probe costs recall credit, however brief. `-guided` pays the larger of its
guided score and what `sequence-timing-behaviour` pays for the same trials.

`sequence-and-timing-recall` scores ten events 20 apart on fields of 240: after the memory recall trigger (a boost of
the input field), LA2B, RA3A, LA1A, RA1B, RA2B, LA3A, RA1A, LA1B, HOL and HOS must first peak in that order, 50, 150,
350, 450, 650, 750, 950, 1050, 1900 and 2800 after it. Each evaluation runs a probe, a demonstration and 1500 without
the trigger; then one demonstration and the trigger five times in turn, each after a wait drawn afresh from 500-1500
(the first from the end of the demonstration, each later one from the end of the recall window before it); then a tail
of 1500 without the trigger. Order is asked of each event: did it first peak at its position, the k-th of the events
that peaked and the k-th of them in the demonstration? Two events first peaking at one sample count as in order, three
or more do not. An event's credit is its best peak score, over three samples running, if it came at its position, and
order is the mean credit times the share of clean samples (at most two events up, each bump at an event and no wider
than a peak, no released event peaking again). Timing, troughs once passed and quiet after the sequence are each paid as
far as the recall was in order. Every recall counts the same, and every recall term is multiplied by the share of
samples with no output bump in the demonstration, the probe and the tail. The guides read the order of the output's
levels at the events: the shown events through the demonstration, the waiting events during recall, and all ten before
the next trigger. Its fitness is the larger of its guided score and the intended behaviour's, whose five weights are in
`config/solutions/sequence-and-timing-recall-intended.json`. A run stops on target only when the fitness and every term
of the larger score, each recall term at its worst recall, exceed it, over six confirmations. A saved solution replays
triggers after waits of 500, 1500, 1000, 500 and 1500.

The `incremental-recall-*` tasks are an incremental-evolution test: one task per stage of recall behaviour, on four
events E1 (40), E2 (90), E3 (15) and E4 (65), shown at 100, 250, 600 and 1200 on fields of 100. The best design of one
stage seeds the next through `neat-dnfs-inc-evol --template`. `incremental-recall-1-single-order` recalls the order
once:
- each evaluation demonstrates the sequence once, then the output recalls three times, each after the memory recall
  trigger comes after its own wait, one drawn from each third of 300-500;
- after the trigger, the output must peak at each event in order, one peak at a time, release each one and stop, and
  stay without a peak until 3200 after the trigger;
- a peak earns its best three samples running, so a peak up for a single sample earns a third;
- every term is the mean over the three recalls;
- there is no timing term.

Its fitness is the larger of its guided score and the intended behaviour's, as in `sequence-and-timing-recall`. The
guides pay what the output holds below threshold: the stored events, the earliest highest, through the demonstration
and the wait; the events still to come, in order, after the trigger; and troughs after each peak. They give the search
a slope before any design recalls. The intended behaviour's six weights are in
`config/solutions/incremental-recall-1-single-order-intended.json`. When the intended behaviour's score is the larger, a
run stops once its six terms pass the target, so a design that recalls without the guides ends the run.

A probe watches the output for 3200 after the demonstration without the trigger: as long as the longest wait of the
later stages and a recall window, so a design that recalls by itself later on is seen. Any peak in the probe costs
recall credit. The three recalls branch off the probe's run: the state at each trigger's moment is saved there, and
restored with its clock for the recall, so the demonstration and the waits are stepped once. A saved solution replays
the trigger after a wait of 300.

`incremental-recall-2-single-order-and-timing` is stage 1 with a timing term: each event's first peak after the
trigger should come at its demonstrated time (100, 250, 600 and 1200), credited as in `sequence-and-timing-recall`.
Its nine guided weights are in `config/solutions/incremental-recall-2-single-order-and-timing.json` and its seven
intended ones in `config/solutions/incremental-recall-2-single-order-and-timing-intended.json`.

`incremental-recall-3-repeatable-order` is stage 1's order-only score with its three waits drawn from the thirds of
300-1500, so the memories must hold through a long wait as well as a short one. A saved solution replays the trigger
after a wait of 1500. Its weights are in `config/solutions/incremental-recall-3-repeatable-order.json` and
`config/solutions/incremental-recall-3-repeatable-order-intended.json`.

`incremental-recall-4-repeatable-order-and-timing` is stage 2's score, timing included, on stage 3's waits and replay.
Its weights are in `config/solutions/incremental-recall-4-repeatable-order-and-timing.json` and
`config/solutions/incremental-recall-4-repeatable-order-and-timing-intended.json`.

`incremental-recall-5-consecutive-order` is stage 1's order-only score on three triggers in one run: the first after
its wait from the end of the demonstration, each later one after its wait from the end of the recall window (1700)
before it, the waits 300, one drawn from 700-1100 and 1500, so every evaluation tries the shortest and
the longest. Each evaluation recalls after three orders of the same three
waits, each wait first, second and third once (one of the two such sets of orders, drawn afresh each evaluation):
nine recalls. A saved solution replays triggers after 300, 900 and 1500 in turn. An event's place in the order is the
share of the other three events on the right side of it, so both events of a swapped pair lose a third. Its peaks must stand still: a peak's drift is its path over its last five
moves without the largest (a peak may split for a sample as it is released), so a bump rolling slowly round the field
peaks at no event it passes. Its one-at-a-time term also pays only as far as every bump on the output stands still at
an event until the sequence is released (each recall's share is `stillAtEvents` in the evaluation trace), and loses a
third of a recall's credit for each sample two peaks are up together (`sharedSamples` in the trace). Its release term
loses a third of a recall's credit for each sample an event peaks again after its release, until the next trigger
(`refiredSamples` in the trace). Its target asks each term of every recall, not their mean (each recall's terms are
`terms` in the trace), so one recall out of order, overlapping or peaking again keeps a design below it. Its weights are in `config/solutions/incremental-recall-5-consecutive-order.json` and
`config/solutions/incremental-recall-5-consecutive-order-intended.json`.

The incremental-recall configs set `PopulationConstants.targetConfirmationEvaluations` to 3 (stage 5's to 6): a run
stops only if its best, evaluated that many more times, reaches the target every time, so one lucky evaluation of a
noisy task cannot end it. Stage 5's confirmations take its two sets of three orders of the waits in turn, so any two
of them recall after all six orders. Absent, it is 0 and the best's own evaluation
decides, as before.

`output-feedback` is not an ablation but a wider search space: mutation may also wire
connections out of output fields (`GenomeMutationConstants.allowOutputAsConnectionSource`), which it
otherwise never does. Its runs go to their own `data/<Task> Output Feedback/` folder. The serial-order
and sequence-timing tasks turn this on in their own config, so they need no preset for it.

`timing-search-levers` is not an ablation either. It is a set of search settings tried on the sequence-timing tasks:
- new fields three times as often (`addFieldGeneProbability` 0.03);
- tau stepping by ×4 rather than ×2;
- resting levels stepping by 0.25 rather than 0.5;
- species and the population each given twice as many generations without improvement before they are cut back (30
  and 50).

Its runs share the task's own data folder; each run's config snapshot records the preset.

## Configuration

Hyperparameters are loaded at startup, not compiled in. `config/neat_dnfs.json` is the
complete reference set; `config/solutions/<task>.json` is that task's *sparse* override of
it, holding only the values that differ for that experiment — its fitness weights, and
anything else it needs (a different `DimensionConstants.xSize`, say, or a different
`AblationConstants.referenceHiddenFieldsMin`). The two are deep-merged and the merged
result is validated strictly, so a missing key or a wrong-length weight array is a startup
error rather than a silently wrong run.

Ablation presets live in `config/ablations/` and are the third layer, merged over the first
two when `--ablation` is passed; adding a file adds a preset, with no code change. A preset
is sparse like a solution override, and may carry a `SolutionConstants` block to hold the run
protocol steady across conditions. Where a preset needs a value that is a property of the
experiment rather than of NEAT, it names the reference instead of hardcoding a number —
`"seedHiddenFieldsMin": "referenceHiddenFieldsMin"`.

Full precedence, later wins:

```text
config/neat_dnfs.json -> config/solutions/<task>.json -> config/ablations/<name>.json -> CLI flags
```

See the [First Steps wiki page](https://github.com/Jgocunha/neat-dnfs/wiki/First-Steps) for the user-facing guide.

## Examples

List everything a binary accepts:

```bash
neat-dnfs-evol --list
neat-dnfs-evol --help
```

Evolve XOR from scratch with the defaults:

```bash
neat-dnfs-evol --task xor
```

Evolve AND with a smaller population/generation budget, for a quick local check:

```bash
neat-dnfs-evol --task and --runs 5 --pop 100 --gens 50 --target 0.9
```

Run one ablation condition on a task:

```bash
neat-dnfs-evol --task and --ablation no-crossover --runs 30 --pop 500 --gens 200 --target 0.9
```

Sweep a task across the control and every ablation (bash):

```bash
for a in "" no-growth-io-only no-growth-reference-hidden-field-count no-speciation no-crossover random-initial-topology; do
    neat-dnfs-evol --task and ${a:+--ablation $a} --runs 30 --pop 500 --gens 200 --target 0.9
done
```

Each arm writes to its own `data/<Task>/` or `data/<Task> <Ablation>/` folder, so the
[analysis dashboard](../analysis/) lists every one as a separate experiment with no extra setup.

Continue evolving inhibition-of-return from its checked-in template:

```bash
neat-dnfs-inc-evol --task ior
```

Continue evolving delayed-match-to-sample from a specific saved solution instead of the task's
default template:

```bash
neat-dnfs-inc-evol --task dmts --template "C:/path/to/solution.json" --runs 20
```

Evaluate a loaded solution 5 times and print its fitness each time. For a task that records what each evaluation
observed (`sequence-timing-behaviour*`, `serial-order-behaviour*`), every evaluation's trace follows its fitness as an
`evaluation trace:` line of JSON:

```bash
neat-dnfs-sol-eval --task ior --evals 5
```

Evaluate a specific saved solution instead of the task's default template:

```bash
neat-dnfs-sol-eval --task memory-trace --template "C:/path/to/solution.json" --evals 10
```
