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
| `--template PATH` | inc-evol, sol-eval | Template JSON to load a starting solution from | the task's own template under `templates/` |
| `--runs N` | evol, inc-evol | Number of independent `Population::evolve()` runs | 50 (evol) / 100 (inc-evol) |
| `--pop N` | evol, inc-evol | Population size per run | 1000 (evol) / 500 (inc-evol) |
| `--gens N` | evol, inc-evol | Max generations per run | 200 (evol) / 100 (inc-evol) |
| `--target F` | evol, inc-evol | Target fitness that ends a run early | 0.95 |
| `--evals N` | sol-eval | Number of `evaluate()` calls | 20 |

Available tasks: `and`, `xor`, `detection-instability`, `memory-instability`,
`selection-instability`, `memory-trace`, `dmts`, `ior`.

Available ablations: `no-growth-io-only`, `no-growth-one-hidden`,
`no-speciation`, `no-crossover`, `random-initial-topology`.

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
for a in "" no-growth-io-only no-growth-one-hidden no-speciation no-crossover random-initial-topology; do
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

Evaluate a loaded solution 5 times and print its fitness each time:

```bash
neat-dnfs-sol-eval --task ior --evals 5
```

Evaluate a specific saved solution instead of the task's default template:

```bash
neat-dnfs-sol-eval --task memory-trace --template "C:/path/to/solution.json" --evals 10
```
