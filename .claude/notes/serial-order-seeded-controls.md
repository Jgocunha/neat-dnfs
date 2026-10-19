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

## Planned runs (not yet run)

One `neat-dnfs-inc-evol` run per seed, at most two at a time, with the `compat-threshold-4` preset. That is the setting
under which the sequence-timing seeds solved: run 18 at 4.0, against run 17 at 1.5 with the same seed.

```
neat-dnfs-inc-evol --task serial-order --runs 1 --gens 100 --target 0.93 --ablation compat-threshold-4 --template <seed>.dnf
```

The preset's label puts the runs in `data/Serial Order Compat 4/`, so the launch script needs
`TASKDIR="Serial Order Compat 4"`.

The seeds and their replays are in the hand-off bundle under `serial-order/architectures/seeds/` (gitignored scratch),
built by `tools/handbuilt/order_seed.py`.
