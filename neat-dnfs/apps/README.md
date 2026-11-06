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

Available tasks: `and`, `xor`, `detection-instability`, `memory-instability`, `selection-instability`,
`memory-trace`, `dmts`, `ior`, `hri-packaging`, `sequence-timing-behaviour`,
`sequence-timing-behaviour-guided`, `serial-order-behaviour`, `serial-order-behaviour-guided`.

Available ablations: `no-growth-io-only`, `no-growth-reference-hidden-field-count`, `no-speciation`,
`no-crossover`, `random-initial-topology`, `compat-threshold-4`, `output-feedback`,
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
