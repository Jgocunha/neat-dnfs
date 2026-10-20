# Serial order: seeded controls

Sequence timing was diagnosed with seeds: the hand-built design with parts removed (levels L1-L3, b1-b3). Evolution
rediscovered one missing mechanism sometimes, and two or more never. Serial order had no such controls, so it was
not known which of its mechanisms from-scratch evolution fails to find. These seeds fill that gap. Design:
`serial-order-handbuilt-design.md`. From-scratch runs: `serial-order-tabula-rasa-runs.md`.

## The seeds

Each seed is the hand-built design (`tests/fixtures/serial-order-handbuilt.dnf`) with one field removed, together with
its kernel, its noise and its three connections. No seed holds a connection evolution could not make, and each removed
field can be rebuilt from connections it can make (`output-field-cannot-be-a-connection-source.md`). Field ids keep their hand-built numbers; the gap is safe since
`f0d4493`.

Removing the clamp C would take out two mechanisms: C freezes the gradient's traces and also tells the latch that
memory holds something. So the gradient seed removes G only.

### Under the evaluator of 19 October 2026 (evenly spaced demonstration, strict disappearance)

Built from the design with the competition in the hidden selection field S (score 0.994). Fifty evaluations each.
Weighting each peak's credit by its height (`f8a1098`) leaves all three unchanged (10 evaluations: 0.358, 0.827,
0.329), since every peak they make is full height.

| Seed | Field removed | What is missing | Fitness | Partials f4, f5, f6 |
|---|---|---|---|---|
| no-done | nf 7 done D | per-item suppression | 0.358 | 0.08, 0, 0.07 |
| no-gradient | nf 5 integrator G | ranking of the stored items | 0.82 (0.77-0.87) | 1, 0.39, 0.99 |
| no-latch | nf 6 recall latch R | lifting the selection field on X | 0.329 | 0, 0, 0 |

The first seeds of this evaluator were built from the earlier design, whose output did the competition and fed the
done field (`gk cg 2-7`). They scored the same: 0.358, 0.80 (10 evaluations) and 0.329. They are kept in the bundle
under `serial-order/architectures/seeds/output-sourced/`. The no-done one was not a fair control: evolution could
never have rebuilt its done field the same way.

The dnf-composer replays (gap 700, cycles 1000, 800, 1500) agree with the evaluator's traces:
- **no-done:** D rises about 35 after the first X at full height (12.0) and never switches off. It is still on when
  the next X arrives, so cycles 2 and 3 see no fresh onset.
- **no-gradient:** all four items rise one at a time and each rises alone (f6 = 1), but in an order set by noise that
  changes from cycle to cycle. Only the ranking is missing, and it is worth the last 0.2.
- **no-latch:** the output never rises.

### Under the earlier evaluator (18-19 October 2026, before `cbbf9f6`)

These seeds were built from the design tuned for the old demonstration (onsets 100, 600, 800, 1800). They are kept in
the bundle under `serial-order/architectures/seeds/old-schedule/`.

| Seed | Fitness | Partials f4, f5, f6 | Behaviour |
|---|---|---|---|
| no-done | 0.357 | 0.083, 0, 0.067 | as above |
| no-gradient | 0.632 | 1, 0, 0.6 | B first, then D, C and A rise together: equal items tied |
| no-latch | 0.328 | 0, 0, 0 | as above |

The retuned design's readout (stronger G->O, lift moved from the latch to memory) separates equal items through noise
instead of letting them tie. That is why its no-gradient seed now recalls one item at a time.

## What scoring ideal behaviours by hand got wrong

Scoring ideal behaviours by hand had put one held winner at 0.41. In serial order, though, a peak that stays on also
costs the next cycles, because no fresh onset can be seen; the no-done seed scores 0.36.

## Runs (20 October 2026)

Two `neat-dnfs-inc-evol` runs per seed, one at a time, under the peak-height evaluator (`f8a1098`), built from
`9800a75`, with the `compat-threshold-4` preset (the setting under which the sequence-timing seeds solved: run 18 at
4.0, against run 17 at 1.5), population 1000, up to 100 generations, `--target 0.93`:

```
neat-dnfs-inc-evol --task serial-order --runs 1 --gens 100 --target 0.93 --ablation compat-threshold-4 --template <seed>.dnf
```

Data: `data/Serial Order Compat 4/`, 01:41-02:08; each run's `config/README.md` has its outcome. Champions and replays
are in the hand-off bundle under `serial-order/runs/sc-run-<n>/`. Re-evaluated is the champion's mean of 10.

| Run | Seed | Stopped at | Reported | Re-evaluated | Outcome |
|---|---|---|---|---|---|
| 1 | no-gradient | gen 18 | 0.991 | 0.828 | Not rediscovered: order still set by noise (order credit 0.49, seed 0.41) |
| 4 | no-gradient | gen 13 | 0.994 | 0.912-0.918 | Partly: A and B always last, D and C sometimes swapped (order 0.71-0.73) |
| 2 | no-done | gen 3 | 0.994 | 0.831-0.884 | Partly: one connection, S to clamp (inhibitory); clean on the first trigger, then degrades |
| 5 | no-done | gen 3 | 0.994 | **0.993** | **Rediscovered:** one connection, S to latch (inhibitory); the latch doubles as the done field |
| 3 | no-latch | gen 15 | 0.960 | 0.941 | Rediscovered, but the output also fires during the demonstration |
| 6 | no-latch | gen 18 | 0.976 | 0.958-0.972 | **Rediscovered:** clean on every trigger, silent demonstration |

**Every one of the three mechanisms was rediscovered.** The done field and the latch within 2-18 generations, the
primacy gradient within 40-68 once the run was not stopped early. Given 100 generations, all three no-early-stop
replicates reached 0.985-0.998 re-evaluated, two of them above the hand-built design's 0.994 (table below).
- **No-done, run 5:** the seed plus one inhibitory connection from the selection field S to the recall latch R
  (amplitude -18, width 8.7). Each winner switches the bistable latch off at its own position for good, so it loses
  its lift and the next item wins; the next X lights the whole latch again. That design has 7 fields, one fewer than
  the hand-built one, and scores as well.
- **No-done, run 2:** the same idea through the clamp, which nothing re-lights, so recall uses up the order.
- **No-latch, runs 3 and 6:** the clamp, on once memory holds something, takes over the latch's lift of S (4-8).
- **No-gradient:** run 1 found nothing before it stopped; run 4 added two fields on a new path from memory to S and
  got a partial ranking. Without early stop (1b below) it found a full gradient.

**`--target 0.93` stops a seeded run on its first evaluation with every term above 0.93.** Single evaluations are
noisy, so every run stopped early, and reported scores ran up to 0.17 above re-evaluated ones. For the no-gradient
seed it is worse: without a ranking, the recall order is set by noise once per trial and repeated on every trigger, so
in a population of 1000 some genome draws D, C, A, B in its one evaluation every generation. One extra replicate per
seed with no early stop (`--target 1.0`, 100 generations) was run alongside, to see what evolution reaches when it is
not stopped by luck:

| Run | Seed | Re-evaluated | Outcome |
|---|---|---|---|
| 1b | no-gradient | **0.998** (10 of 10) | **Rediscovered**, above the hand-built 0.994: the clamp field nf 4 holds a primacy gradient (D 184, C 178, A 177, B 172.5 during recall); full peaks of 17.5, one at a time, silent demonstration. 11 fields, 22 connections |
| 2b | no-done | **0.998** (10 of 10) | **Rediscovered**, above the hand-built 0.994: the same inhibitory S-to-latch connection as run 5 (8-6, -13.8), plus refinements; full peaks of 11, one at a time, silent demonstration. 8 fields, 19 connections |
| 3b | no-latch | **0.985-0.995** | **Rediscovered:** clean recall, silent demonstration, but slow (B about 800 after X), so at the shortest cycle (800) B runs into the next one. 7 fields, 15 connections |

The stop rule needs re-evaluation before it stops a run, for example the best genome's mean over a few evaluations;
that is an evolution-loop change, not made.

The seeds and their replays are in the hand-off bundle under `serial-order/architectures/seeds/` (gitignored scratch),
built by `tools/handbuilt/order_seed.py`.
