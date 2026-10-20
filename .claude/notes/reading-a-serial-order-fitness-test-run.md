# Reading a serial-order-fitness-test run

A run of `neat-dnfs-evol --task serial-order-fitness-test` writes into `data/Serial Order Fitness Test/<start time>/`.

## What each file holds

- `overview.jsonl`: one record per generation g, starting at 0. Each record holds:
  - the fitness distribution;
  - each species' id and size;
  - the best solution: id, parents, partial fitnesses and its `evaluationTrace`.
- `statistics/generation_{g+1}.txt`: one line per solution of generation g. Each line holds:
  - fitness, partial fitnesses and species;
  - the parents, fitter first;
  - the genome's field and connection ids;
  - `last mutations{...}`, the mutations that made it.
- `species/generation_{g+1}.txt`: one line per species, with:
  - its age, members, offspring and generations since it improved;
  - its representative and champion, as in the statistics lines.
- `best_solutions/prev_generations/` and `champions/prev_generations/`: one `.dnf` per generation for the best
  solution and for each species' champion, with full parameters. They are named with g+1. Each carries the
  evaluation schedule, so it replays the evaluation in dnf-composer.
- Run with `NEAT_DNFS_DATA_DIR=<dir>`, everything goes under `<dir>/data/` instead, which suits smoke tests.

The files are numbered one ahead of the overview: the overview's generation g is `generation_{g+1}`.

## Tracing an offspring

- Champions and the global best are carried into the next generation unchanged, under the same id.
- Every other solution of generation g is new: it is crossed from two parents of generation g-1, fitter first, and
  then mutated.
- So an offspring's fitness change is its fitness minus its fitter parent's in the previous file.
- The mutation string names every operator that fired. For example, `(cg to inhibitory)` is a sign flip, and
  `(fg nf rand.)` is a field re-draw.

## The evaluation trace

There are 27 snapshots: 9 in the demonstration and 6 in each recall cycle. Their stages are:
- `before D`;
- `<event> shown` at the end of each pulse;
- `after <event>` at the next onset;
- `recall <cycle> <event>` at each recall point;
- `recall <cycle> end` at 600 after X;
- `recall <cycle> rest` at the cycle's end.

Each snapshot holds:
- `t`;
- the input's `bumps`, as [centroid, amplitude, width];
- the output's `bumps`, `median` and `atEvents`, its activation at D, C, A and B.

Values are rounded to two decimals. To read memory and suppression, subtract the median from `atEvents`: positive
means pre-shaped, negative means a trough.

`neat-dnfs-sol-eval --task serial-order-fitness-test --template <file.dnf>` logs the same trace after each
fitness line, as `evaluation trace: {...}`, for any saved genome.
