# Serial order: what holds evolution back, and which levers to pull

21 October 2026. Sources:
- the 24 completed from-scratch runs since 15:00 on 20 October (batches w, g, h, t, o; 4.56 M offspring);
- their 30 final bests, re-scored at `f2ea25d`;
- variants of the reference, `tests/fixtures/serial-order-fitness-test-reference.dnf`.

Scripts: `.claude/temp/fitness-test-runs/tools/population_dynamics.py` and `tau_usage.py`.

## Bottom line

1. **Selection is close to random.** Species flip between one species of 1000 and 100-500 species of a few members.
   No species but the founder lives 5 generations, and the best genome parents 0.2% of the offspring: chance.
2. **Offspring change about 4.5 parameters at once.** Each extra change adds damage and no benefit.
3. **The input and output are drawn at random and mutate freely.** So evolution gets memory cheaply:
   - a slow output, in 17 of 30 bests;
   - an input that stays active after its pulses, in 6.

   Both are dead ends for recall: the reference with a slow output scores 0.391.
4. **The missing structure is the release loop (the done field).** Part of it appears in 7 of 30 bests; its reset at X
   in none. A partial loop scores no more than no loop. Only the whole loop pays (0.96).
5. **Every building block is within the ranges.** Mexican hats are not needed: a Gauss-only reference scores 0.96.

## 1. The search

### Speciation protects nothing

- `Species::isCompatible` divides the genetic distance by N, the larger connection-gene count, but sets N = 1 below 20
  genes.
  - Our genomes carry 13-21 connection genes, disabled ones included.
  - So the distance shrinks 20-fold as soon as one of the two genomes reaches 20 genes.
- In every run:
  - one species holds at least 90% of the population in 40% of generations;
  - in 95% of those, that species' representative has 20 or more genes;
  - otherwise there are a median of 110-160 species in most runs, and up to 549;
  - apart from the founder, no species lives past 4 generations (median 0).
- The distance counts connection genes only. A field's tau, resting level or kernel never separates two genomes, so a
  field that turns slow is never protected.
- **Compatibility coefficients alone cannot fix this.** Any threshold either splits the small genomes into hundreds or
  merges the large ones into one. It needs the 20 made configurable (or N = 1 always), then a threshold tuned to 20-50
  species. That is a small code change.

### Selection is almost random

- A species gets offspring in proportion to its mean fitness, so 130 species of similar means get about 8 each.
- Inside a species, parents are drawn uniformly from the best 80% (`pruneRatio` 0.2). Species of 4 or fewer are not
  pruned at all.
- **Measured:**
  - the fitter parent averages 0.4-0.5 standard deviations above the population mean in the middle half of runs (0.2-0.7
    overall), and 0.1-0.3 in the first 50 generations;
  - the previous generation's best parents 0.2% of the offspring, exactly chance.
- **`pruneRatio` is the dial.** Selection intensity within a species is about 0.35 SD at 0.2, 0.8 at 0.5, and 1.4 at 0.8
  (the base config's value).

### Too many changes per offspring

0.5 × 0.4 per gene, at 7 fields and 18 connections, gives 4.5 changes per offspring (median 3), and about 40% of
offspring carry 5 or more. The number grows with the genome.

| Changes | Clone: worse than parent by > 0.05 | Clone: mean change | New population best, clone / crossover |
|---|---|---|---|
| 0 | 2.3% | -0.001 | 0.21% / 0.17% |
| 1 | 9.8% | -0.014 | 0.15% / 0.19% |
| 2 | 12.4% | -0.019 | 0.14% / 0.14% |
| 3-4 | 15.5% | -0.024 | 0.12% / 0.09% |
| 5-8 | 19.9% | -0.032 | 0.09% / 0.05% |
| 9+ | 27.7% | -0.048 | 0.05% / 0.03% |

- **Single changes, by kind.** The most damaging are:
  - a field redraw, worse in 49% of cases;
  - a connection sign flip, 40%;
  - toggling a connection, 35%;
  - a kernel-type flip, 32-35%.

  Tau steps do best: better in 8.7% of cases and worse in 12.8%, where other parameter steps are better in 7.3-8.1%.
- **Crossover is damaging on average.** With no mutation, a crossover is worse than its parent by more than 0.05 in 14%
  of cases, against 2% for a copy. In merged generations it crosses unrelated architectures.
  - Yet crossovers without any mutation made 35% of all new population bests, so recombination does real work.
  - Fix speciation before judging it.
- **Two quirks of the code:**
  - one kernel mutation moves width, amplitude and global amplitude with a single shared sign;
  - one field mutation moves tau and resting level together the same way.

  So "more excitation and more global inhibition" can never happen in one step.

### The stagnation rules rarely act

- They fire 0-3 times per run for the population (offspring to the top two species) and 0-33 times for species.
- Species die young, so the species threshold only ever reaches the founder.
- Lowering either would mostly add collapses to two species. Leave them until speciation works.

### Noise is a tail

- 1 re-evaluation in 10 moves by more than 0.01-0.02, and 1 in 100 by more than 0.08-0.11.
- 10% of new population bests were unchanged copies of a parent, so they are lucky re-evaluations.
- Final bests are stable, except w1 (0.560-0.675). This is secondary.

## 2. The input and output fields

- They are built like hidden fields:
  - tau log-uniform over 5-10,000 (median 224);
  - resting level uniform over -20 to +1;
  - kernel Gauss or Mexican hat;
  - all of it mutable.
- In generation 1, the bests already had outputs at tau 1,000-9,000 in 7 of 24 runs, and output resting levels of +0.5
  or more in 10.
- **Evolution uses this:**
  - 17 of 30 bests keep their memory in a slow output (tau 310-2,800). Setting it to 10 costs 0.02-0.38.
  - o1, o2, o5, g1, g2 and h4 keep the input active after its pulses (f1 0.40-0.66).
    - In o1 that is four self-sustained input peaks.
    - Four of the six inputs have Mexican hats, with resting levels from -5.4 to +0.5.
- **Both block recall.** A slow output cannot step or release in time: the reference with its output at tau 500 scores
  0.391 (f4 0.13).
- **Fix:** give the input and output fixed parameters, as part of the design.
  - The input as the reference's detection field: tau 10, resting level -10, weak Gauss.
  - The output at tau 10, with its resting level and kernel left to evolve.

  `NeuralFieldConstants` apply to every field, so this needs a small code change.

## 3. Tau

- **How a high tau is reached.** Only by a redraw, or by inheritance:
  - a redraw is log-uniform, and 30% of redraws land at 1000 or more;
  - a new hidden field copies its source field's tau.
- **Steps cannot get there.** A ×1.3 step is benign but rare: 0.54% of fields per generation. Going from 10 to 10,000
  takes 26 of them. `tauStep` is unused while `tauMutationFactor` is set.
- **A redraw also redraws the resting level, over -20 to +1.** Half of redraws hurt.
- **Is it used?** 24 of 30 bests have a field at tau 300 or more, and 11 at 1000 or more. None is above 9,786.
  - Mostly it is the output (above).
  - A separate slow memory field, as in the reference, appears in a few:
    - t6 (tau 2,779; -0.37 when set to 10);
    - t4 (9,786; -0.27);
    - t9, g3, g4 and h6.
- **The reference needs it:** with nf 5 at tau 10, it scores 0.368.
- **Config:**
  - `tauMutationFactor` 2, so 10 to 10,000 takes 10 steps;
  - tau-step share 0.15 → 0.4, and resting-level share 0.5 → 0.2, since resting-level steps matter less;
  - redraw 0.1 → 0.05.

## 4. Mexican hats

- **The reference uses them on two fields, never on a connection.** They are the self-kernels of nf 3 and the done
  field nf 7, the two self-sustaining fields.
- **A naive Gauss swap breaks it** (0.30-0.38). But Gauss amplitude 9-13, width 1-1.75 and global 0 to -0.05 score
  0.95-0.96 on every evaluation: 13 of 70 settings, all inside the ranges.
- **In the bests,** 80 of 205 field kernels and 42 of 398 connection kernels are Mexican hats.
- **Config:** Gauss only (`gaussKernelProbability` 1.0, `mexicanHatKernelProbability` 0.0) for connections and fields.
  - Add the Gauss-only reference as a fixture first, so the "ranges reach the reference" test still covers it.
  - With Gauss only, a kernel-type mutation just redraws a Gauss kernel, one of the most damaging changes. Move its share
    to kernel steps: fields 0.7 + 0.1, connections 0.93 + 0.02.

## 5. The missing behaviour, and why

| Genome | f1 input | f2 quiet | f3 memory | f4 recall | f5 to come | f6 release | f7 stop |
|---|---|---|---|---|---|---|---|
| reference | 1.00 | 1.00 | 1.00 | 0.96 | 0.95 | 0.93 | 0.98 |
| o3 (burst) | 0.96 | 1.00 | 1.00 | 0.45 | 0.91 | 0.22 | 0.50 |
| t6 | 0.90 | 0.78 | 0.60 | 0.47 | 0.86 | 0.10 | 0.00 |
| o5 | 0.50 | 0.73 | 0.20 | 0.75 | 0.27 | 0.31 | 0.17 |
| o1 | 0.49 | 0.39 | 0.08 | 0.84 | 0.26 | 0.31 | 0.08 |
| o4 | 0.96 | 0.73 | 0.68 | 0.35 | 0.84 | 0.04 | 0.00 |
| t1-t10 | 0.86-0.99 | 0.76-1.00 | 0.61-1.00 | 0.09-0.23 | 0.76-0.98 | 0.00 | 0.00 |

- **Found:**
  - ordered pre-shapes;
  - a detecting input;
  - the output's competition: a global inhibition of -0.6 to -1.2 in 14 of 30 bests.
- **Found only with faked memory:** recall one peak at a time (o1, o5).
- **Never found:** release and stop.

**The reference's release loop.** Done field nf 7:
- driven by the output (2-7, +8);
- inhibiting it (7-2, -30);
- self-sustaining (Mexican hat 20/12);
- reset at every X by the input's boost (1-7, global -0.2).

Removing parts of it:

| Variant | Fitness | f4 | f6 | f7 |
|---|---|---|---|---|
| no loop | 0.482 | 0.24 | 0.00 | 0.00 |
| loop, no self-excitation, no reset (adaptation) | 0.549 | 0.38 | 0.09 | 0.00 |
| loop, no self-excitation | 0.507 | 0.27 | 0.07 | 0.00 |
| loop, no reset | 0.474 | 0.32 | 0.32 | 0.66 |
| whole loop | 0.960 | 0.96 | 0.93 | 0.98 |

- **The done field's tau hardly matters:** tau 10 still scores 0.945.
- **Other essentials:**
  - competition in the output (without it, 0.377);
  - the slow integrator (nf 5 at tau 10, 0.368);
  - a fast output (at tau 500, 0.391).

**Why it is missing:**
- **A valley.** From the adaptation loop, adding the done field's memory alone (0.474) or its reset alone (0.507) loses
  fitness. Only both together pay, and the search cannot carry a two-step innovation across a valley: nothing lives past
  4 generations.
- **Cheaper substitutes.** Memory through a slow output or a self-sustaining input pays f3 and f5 at once, and makes the
  release harder.

**Building blocks.** All are within the ranges and legal:
- output as a connection source;
- inhibition down to -50;
- global amplitude ±0.8;
- tau up to 10,000.

**Valuing the loop more, by behaviour only.** f7, the stop, is the behavioural sign of a done memory. Moving weight to
it removes the valley:

| Genome | now (.05 .05 .10 .35 .20 .20 .05) | f7 0.20 (.05 .05 .10 .30 .10 .20 .20) | f7 0.30 (.05 .05 .10 .25 .05 .20 .30) |
|---|---|---|---|
| no loop | 0.484 | 0.372 | 0.310 |
| adaptation loop | 0.549 | 0.431 | 0.363 |
| + memory, no reset | 0.475 | **0.525** | **0.558** |
| whole loop | 0.961 | 0.965 | 0.968 |
| o3 burst | 0.607 | 0.568 | 0.550 |
| t6 | 0.501 | 0.391 | 0.325 |

- **The trade-off:** f7 pays half for an output with no peak, so a silent genome rises from about 0.12 to about 0.20 at
  f7 0.20. Gating that half on recall, as the troughs already are, would avoid it.

**A direct test of findability.** `neat-dnfs-inc-evol --task serial-order-fitness-test --template <reference without
2-7, 7-2, 1-7>` starts every run on the 0.482 plateau. It shows whether NEAT, as configured, can build the release at all.

## 6. Proposed order

| # | Change | Kind | Expected effect |
|---|---|---|---|
| 1 | Configurable small-genome cutoff (or N = 1 always); threshold tuned for 20-50 species | small code + config | species that last; innovations protected |
| 2 | Fixed input (detection field) and fast output | small code | removes the two dead-end memories |
| 3 | Per-gene mutation 0.4 → 0.15 (fields), 0.10 (connections) | config | about 1.4 changes per offspring instead of 4.5 |
| 4 | `pruneRatio` 0.2 → 0.5 | config | selection intensity about 0.8 instead of 0.35 |
| 5 | `tauMutationFactor` 2; tau 0.4, resting level 0.2, redraw 0.05 | config | slow fields reachable by steps |
| 6 | Gauss only; kernel-type share to kernel steps; Gauss-only reference fixture | config + test | fewer damaging flips; reference still reachable |
| 7 | f7 weight up (e.g. 0.20), its no-peak half gated on recall | fitness | removes the release valley |
| 8 | inc-evol from the 0.482 plateau | diagnostic | whether the release is findable at all |

Leave alone:
- the stagnation thresholds;
- simulation, dimension, noise and stimulus constants;
- kernel constants;
- Gauss kernel ranges (the Gauss-only reference's self-kernels sit inside them).

## What changed (21 October 2026)

| Commit | Change | Serial-order value |
|---|---|---|
| `dc27217` | `CompatibilityCoefficients.minConnectionGenesToNormalize`, optional, default 20 | 100 (never normalise) with threshold 6, in both serial-order configs |
| `5126d7c` | `FieldGeneConstants.fixedInputField` / `fixedOutputField`, optional | input tau 10, resting level -10, Gauss 2/3/0; output tau 10 |
| `a2eba57` | `GenomeMutationConstants.independentStepSigns`, optional, default off | on |
| `ed522c0` | per-gene mutation, `pruneRatio`, tau mutation, Gauss only | fields 0.15, connections 0.10; 0.5; see below; Gauss 1.0 |
| `a892c59` | f7's no-peak half counted as far as the cycle recalled; weights | [.05 .05 .10 .30 .10 .20 .20] |

**Threshold 6.** Replaying `Population::assignToSpecies` on saved populations with N = 1 throughout gives a median of
22-52 species across generations at threshold 6, and 10-28 at 8.

**Tau: the proposal above was wrong in combination.** With per-gene field mutation at 0.15, every field change is 2.7
times rarer, redraws included, and only redraws reach high tau. Simulated without selection, a field born at tau 10 that
reaches 1000 within 200 generations:

| Setting | Share reaching 1000 within 200 generations |
|---|---|
| before (field 0.4, NF 0.2, redraw 0.1, tau 0.15, ×1.3) | 22% |
| section 3 as written (field 0.15, NF 0.2, redraw 0.05, tau 0.4, ×2) | 4.6% |
| applied (field 0.15, NF 0.4, redraw 0.1, tau 0.4, ×2, resting level 0.2) | 18% |

The applied setting keeps redraws near their old rate, and makes tau steps twice as frequent and twice as large.
Resting-level steps are 3 times rarer.

**The release valley, re-scored with the new f7:**

| Variant | Fitness |
|---|---|
| no loop | 0.370 |
| adaptation loop | 0.431 |
| + memory, no reset | 0.456 |
| + reset, no memory | 0.394 |
| whole loop | 0.960 (Gauss-only 0.963) |

- The path through the done field's memory is now uphill all the way.
- Adding the reset first is still downhill.
- A slow output scores 0.321.

**Gauss-only reference.** `tests/fixtures/serial-order-fitness-test-reference-gauss.dnf`: nf 3 and nf 7 with Gauss
self-kernels, amplitude 10 and width 1.25. The ranges test now checks it.

**Diagnostic template.** `.claude/temp/fitness-test-runs/plateau-without-release-loop.dnf` is the Gauss-only reference
without nf 7 or any of its connections. It scores 0.370.

## Smoke run: the start collapses (21 October 2026)

`neat-dnfs-evol --task serial-order-fitness-test --runs 1 --pop 1000 --gens 30` on `a892c59`, data in
`.claude/temp/smoke-search/`.

| Measure | Expected | Measured |
|---|---|---|
| Species | 20-50 | 1, every generation |
| Changes per offspring | about 1.4 | 0.11; 91% of offspring unchanged |
| Fields / connections at generation 29 | growing | 2.0 / 1.0 (old runs: 4.9-5.9 / 9.3-12.4) |
| Best | rising | 0.126, flat from generation 6 |
| Input and output | fixed | fixed: nf 1 tau 10, resting level -10, Gauss 2/3/0; nf 2 tau 10 |

- **Both calibrations used the old runs' genomes.** Changes per offspring and genetic distance both grow with the genome,
  and the old runs' genomes had about 5 fields and 10 connections. The starting genome has 2 fields and 0-1 connections,
  and every parameter of the input is fixed, so its mutations change nothing.
- **No static threshold fits a whole run.** Thresholds that give 20-50 species on saved populations:

  | Population | Connections | Threshold |
  |---|---|---|
  | smoke run, generations 5-30 | 0.6-1.0 | none: even 0.5 gives 2-8 |
  | old runs, generation 10 | 6-7 | 3-4 |
  | old runs, generation 30 | 10-11 | 8 |
  | old runs, generation 100 | 15-18 | 6 |
  | old runs, generation 200 | 19 | 1.5-3 |

  The old threshold, 1.5, also split identical relays. The same connection re-added in another generation gets a new
  innovation number, and two such relays lie 1.5 apart.
- **The best holds D.** From generation 6 the best is the bare input-to-output connection. The output (self-kernel
  amplitude 18.6, resting level -2.26) ignites at D during the demonstration and holds the peak to the end.
  - f4 counts a held D as D recalled first, in every cycle: 0.24, worth +0.073. Holding costs f2 only 0.047.
  - The bests of generations 4 and 5 had no connections at all: a peak ignited from noise near 180 and was held, and
    scored the same way.
- **Every added field lost.** 257 offspring gained a field. 99% scored worse, by a median of 0.0255, which is the held D
  lost (-0.073 + 0.047). None survived.

Proposed, not yet approved:
1. **Species:** a threshold that steps each generation towards a target species count (NEAT's dynamic threshold).
   Optional, off by default.
2. **Mutation:** every offspring changes at least one parameter. Optional, off by default.
3. **f4:** an event counts as recalled in a cycle only as far as it was not already a peak when that cycle's X came.
