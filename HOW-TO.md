# HOW-TO

A practical guide to running experiments and changing parameters. If you only want to
tune the framework, the two files you edit are:

- `neat-dnfs/config/neat_dnfs.json` — the reference hyperparameters
- `neat-dnfs/config/solutions/<task>.json` — what differs for one experiment

Nothing else needs touching, and no source edit or rebuild is required to change a value.

---

## 1. Run an experiment

Three executables, all sharing the same tasks and flags:

| Binary | What it does |
|---|---|
| `neat-dnfs-evol` | Evolves a task from a minimal starting topology |
| `neat-dnfs-inc-evol` | Evolves onward from a saved solution in `templates/` |
| `neat-dnfs-sol-eval` | Re-evaluates a saved solution, without evolving |

```bash
neat-dnfs-evol --task xor
neat-dnfs-sol-eval --task ior --evals 5
```

Available tasks: `and`, `xor`, `detection-instability`, `memory-instability`,
`selection-instability`, `memory-trace`, `dmts`, `ior`.

`--list` prints every task and ablation; `--help` prints every flag.

Results are written to `neat-dnfs/data/<solution name>/`.

## 2. Run an ablation

```bash
neat-dnfs-evol --task xor --ablation no-crossover
```

Available ablations: `no-growth-io-only`, `no-growth-reference-hidden-field-count`,
`no-speciation`, `no-crossover`, `random-initial-topology`.

## 3. Change hyperparameters

**`neat-dnfs/config/neat_dnfs.json` is the reference set** — mutation rates, kernel
defaults, field size, population control, and the run protocol. Edit it to change a value
everywhere.

**`neat-dnfs/config/solutions/<task>.json` is that experiment's override.** It only lists
what differs from the reference, so it stays short. This is the point of the split: a task
that needs a larger field or a longer run keeps its own working values, and you always have
a saved parameter set that works for it.

```jsonc
// neat-dnfs/config/solutions/xor.json
{
  "SolutionConstants": {
    "fitnessWeights": [0.25, 0.25, 0.25, 0.25],
    "populationSize": 500,      // this task runs a smaller population
    "numberGenerations": 150
  },
  "DimensionConstants": {
    "xSize": 200                // ...on a wider field than the reference
  }
}
```

Keys and nesting match `neat_dnfs.json` exactly — copy a block across and change the number.

### Which value wins

Later layers override earlier ones:

```
config/neat_dnfs.json  ->  
config/solutions/<task>.json  ->  
config/ablations/<name>.json  ->  
CLI flags reference this experiment only when --ablation is used highest
```

So `--pop 200` beats everything, and an ablation's population size beats the task's.

### Run protocol

`populationSize`, `numberGenerations`, `numberRuns` and `targetFitness` live in
`SolutionConstants`, because how big a population a task needs — and what score counts as
solved — is a property of that task. Each has a CLI equivalent for one-off changes:
`--pop`, `--gens`, `--runs`, `--target`.

## 4. Add an ablation

Drop a JSON file into `neat-dnfs/config/ablations/`. The filename is the name you pass to
`--ablation`, and it appears in `--list` automatically. No code change, no rebuild.

A preset is sparse — list only what the condition changes:

```json
{
  "AblationConstants": {
    "label": " No Crossover",
    "disableCrossover": true
  },
  "SolutionConstants": {
    "populationSize": 500,
    "numberGenerations": 200,
    "numberRuns": 30
  }
}
```

The `SolutionConstants` block is optional; include it to hold the run protocol steady across
conditions so results are comparable. `label` is appended to the output folder name.

Where a preset needs a value that depends on the task rather than on NEAT, name the reference
instead of hardcoding a number:

```json
"seedHiddenFieldsMin": "referenceHiddenFieldsMin",
"seedHiddenFieldsMax": "referenceHiddenFieldsMax"
```

Those resolve against `AblationConstants.referenceHiddenFieldsMin`/`Max`, which each task can
set for itself — so `no-growth-reference-hidden-field-count` seeds however many hidden fields
*that* experiment treats as its reference, not a number that only suits one task.

## 5. What not to touch

- **`include/constants.h`** holds only declarations. Change
  `neat_dnfs.json` instead.
- **`src/solutions/*.cpp`** holds each task's stimulus positions and target bump geometry.
  That is deliberate — it defines what the task *is*, not how it is tuned. Only the fitness
  weights are configurable.

## 6. If a run refuses to start

Configuration errors fail immediately with a message naming the file and the key, rather than
running with a wrong value. Expect a startup error if:

- a config file is missing or is not valid JSON
- a key is missing from `neat_dnfs.json` (the reference must be complete; only the override
  files may be sparse)
- `fitnessWeights` has the wrong number of entries for the task
- a preset names a reference that does not exist

## 7. Reproducibility note

Evaluation is stochastic (`NoiseConstants.amplitude`) and the RNG is not currently seedable, so
repeated runs differ. To compare two configurations exactly, set `NoiseConstants.amplitude` to
`0.0` in a copy of the config and pass it with `--config`:

```bash
neat-dnfs-sol-eval --task xor --evals 3 --config my_no_noise.json
```

With noise off, evaluating a fixed solution is deterministic and repeatable.
