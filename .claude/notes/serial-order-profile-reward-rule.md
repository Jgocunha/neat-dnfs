# Serial-order profile: which reward rule (21 October 2026)

The `serial-order-profile` task scores a design by how closely the activation of its input (`nf 1`) and output (`nf 2`)
fields through the whole serial-order trial matches the Gauss-only reference design's. Both are recorded every 10 at all
200 positions, from 0 to 5850: 586 samples per field (`config/references/serial-order-profile.json`, written by
`neat-dnfs-record-profile`). This note records how the rule that turns the comparison into a fitness was chosen.

## Test set

196 designs, each recorded with the task's own code path:

- **The reference.** Three repeat recordings of the reference set the noise floor. The original reference, with
  Mexican hats in two hidden fields, behaves almost the same.
- **The 30 replays of 20 October** (`.claude/temp/fitness-test-runs/replays`). The user judged 21 of them: o1, o4, o5
  and t6 were closest to solving the task, and the g and h runs were bad.
- **Failure-mode fixtures** (`tests/fixtures/serial-order-*.dnf`).
- **The 21 October smoke run's best**, a relay that holds a peak at D.
- **Built from the reference:**
  - silent: the input and the output with nothing connected, at several resting levels;
  - the reference without each connection, and without each hidden field;
  - one-parameter sweeps;
  - every connection scaled by 0 to 1.1;
  - 60 random starting-like designs: the fixed input, an output drawn from the task's ranges, and two in three with a
    random input-to-output connection.
- **Synthetic profiles** made from the reference's own output: shifted in time or place, offset in level, with its
  features scaled.

Scripts: `.claude/temp/profile-investigation/` (`make_designs.py`, `record_all.py`, `analyse*.py`).

## What a rule had to do

- Score the reference about 1, above noise.
- Rank the designs the user judged good above the bad ones.
- Not let a silent output beat designs that do part of the task.
- Not be dominated by the field's overall level.
- Fall gradually with shifts in time and place, and with partial mechanisms.
- Rise along the path from silent to the reference.
- Peak at the reference's value in every one-parameter sweep.
- Weight nothing explicitly, peaks included.

## Results (output field; the fixed input scores the same for every new design)

| Rule | Silent | Judged above silent | o,t above g,h | Favourites above the rest | Level only | Reference offset −9 |
|---|---|---|---|---|---|---|
| Mean absolute difference, 1/(1+MAE/MAD) | 0.46 | 14% | 0.77 | 0.58 | 0.73 | 0.29 |
| Mean squared difference (NNSE) | 0.33 | 14% | 0.75 | 0.68 | 0.61 | 0.18 |
| Pearson / CCC over the whole profile | 0.50 | 86% | 0.41 / 0.62 | 0.47 / 0.50 | 0.83 | 1.00 / 0.65 |
| SSIM | 0.79 | 14% | 0.72 | 0.44 | 0.90 | 0.87 |
| **Concordance at each moment, regularised (k = 1)** | **0.52** | **100%** | **0.92** | **0.71** | **0.57** | **0.59** |

The "o,t above g,h" and "favourites above the rest" columns are shares of pairs ranked the right way (AUC).

**Plain point-by-point difference fails.** The reference's output sits at −13 during the demonstration and recall, but
its whole field rises to −4 for about a quarter of the trial (between recalls and at the start). That level change
outweighs every feature:

- A profile that copies only the level and has no pre-shape, peak or trough scores 0.73.
- A perfect copy offset by −9 scores 0.29, below a silent output (0.46).
- Silence beats 18 of the 21 judged designs.

The judged designs lose mostly because their outputs rest at other levels: o1's at −21.6.

**The recommended rule** compares the field at each moment with the reference's at the same moment, by Lin's concordance
correlation, regularised for flat moments:

    agreement(t) = 1 - mean((u - u_ref)^2) / (var(u) + var(u_ref) + (mean(u) - mean(u_ref))^2 + 1)
    score(t)     = (1 + agreement(t)) / 2        means and variances over the field's positions at time t

- A field identical to the reference scores 1, an unrelated one 0.5, and an opposite one 0.
- When the reference is flat at a moment (the input between pulses), a candidate at its level scores 1, one unit off
  0.75, and three units off 0.55. Without the +1, every flat moment scores 0.5 even for the reference: its input would
  score 0.55.
- The fitness is the mean of score(t) over every moment and both fields.

How it behaves:

- **Noise floor.** Repeat recordings of the reference score 0.999 and the original reference 0.944. k = 0.1 gives the
  same results; k = 4 starts to favour silence.
- **Shifts in time.** +30 scores 0.95, +100 0.87 and +200 0.78.
- **Shifts in place.** 2 positions score 0.95, 5 score 0.77 and 15 score 0.46. At 15 the features no longer overlap,
  so the design is penalised twice, for the missed feature and for its own, and scores below silence.
- **Level.** +3 scores 0.82 and −9 scores 0.59.
- **Feature height.** Features at half height score 0.90, and at 1.5 times height 0.96.
- **Path from silent to the reference.** 0.52, 0.53, 0.65, 0.74, 0.88, then 1.00.
- **Sweeps.** All 8 one-parameter sweeps peak at the reference's value. The memory field's tau is flat from 10 to 300
  (0.83) and climbs from 1000.
- **Stepping stones:**
  - random starting-like designs score 0.50–0.57;
  - the old runs' memory and held-peak designs 0.55–0.67;
  - the reference without one part 0.55–0.89.
- **The held D of the smoke run scores 0.61.** It is credited through the demonstration, where D is the reference's
  highest pre-shape, and falls below silence from the recall of C on.
- **o1 ranks 17th of 30 at 0.57.** It steps in order, but its output rests 12 below the reference's and its peaks are a
  third as tall.

## Consequences

- **The input term is constant.** The input's parameters are fixed and nothing can connect into it, so every new design
  matches the reference's input (1.00). The fitness is then (1 + output)/2, from about 0.76 for silence to 1.
  - Ranking within a species is unchanged.
  - Offspring shares between species are flatter.
- **Stages.** Stage scores make useful partial terms. The demonstration has six stages: trigger, D, C, A, B and wait.
  Each recall has six, with boundaries at the reference's own peak onsets: X, recall D, C, A and B, then rest. The
  fitness is their duration-weighted mean.
  - Averaging stages equally instead would let a 2-sample X stage count as much as a 97-sample rest.
  - It ranked the judged designs about the same: favourites 0.76 against 0.71, and o,t above g,h 0.89 against 0.92.
  - **Superseded the same day:** the wait and the rests are no longer scored. See "Scoring only the moments when
    something happens" below.

## What evolution found: a design that never recalls scores 0.871

The first 200-generation run from scratch (`neat-dnfs/data/Serial Order Profile/2026-10-21 18h18m17s`) ended on a
design that recalls nothing:

- Through the demonstration it holds peaks at D and C and pushes A and B 5–6 below the field.
- The first recall cue wipes the peaks. From then to the end the output sits at −4.1 with all four places 5–6 below
  it, and no event place crosses threshold again.

Its fitness is 0.871 against 0.76 for silence. The output alone scores 0.743, broken down here by part of the trial:

| Part of the trial | Share | This design | Silent output |
|---|---|---|---|
| Demonstration | 44% | 0.78 | 0.54 |
| X and rests | 31% | 0.94 | 0.51 |
| Recall stages | 26% | 0.45 | 0.52 |

- **Rests.** After recall the reference's output rises to −4, with all four places pushed down. This design copies that
  state, without the recall that leads to it.
- **Demonstration.** The held peaks sit where the reference builds its two highest pre-shape bumps, and concordance
  forgives features that are too tall.

**This breaks a criterion the test set never challenged.** The design's output scores above all 30 replays (best
0.67), including o1 (10 of 12 recalls) and t6 (9 of 12). The replays score better in the recall stages (0.62–0.70),
but they rest at −16 to −23 and lose the rests (0.29–0.45).

Three variants scored on the same profiles fail too:

| Variant | Result |
|---|---|
| Every stage counted the same | The design still beats all four favourites (0.645 against 0.58–0.63). |
| The field minus its median at each moment (shape without level) | The design still beats all 30 replays. |
| The field's output, sigmoid(u), instead of u | The design falls to 37th of 40, but silence (0.90) beats every replay. |

Scripts: `explain_0871.py`, `rank_designs.py` and `rank_by_stage.py` in `.claude/temp/profile-investigation/`.

### A second case: a recall stage scores 0.96 without its peak

Two incremental runs started from the reference with one hidden field removed
(`neat-dnfs/data/Serial Order Profile/2026-10-21 19h18m05s` and `19h18m10s`). They ended in the wrong order:

| Run | Best | Recalls |
|---|---|---|
| Without the memory field | 0.968 | All 12, in order |
| Without the recall driver | 0.970 | B every cycle, A fading, C once and late, D never |

In the second run the output sits at −1 at D through every recall-D window, where the reference peaks at +12. Those
stages still score 0.957–0.965. A missing peak changes only the few positions it covers out of 200, so the concordance
at that moment barely drops.

## Scoring only the moments when something happens

The user asked for the times when nothing happens to be left out. A moment counts if it falls in one of these windows:

- **Each stimulus**, from its onset to 50 after its pulse ends (onset + 100): 0–200, 600–700, 800–900 and 1800–1900.
- **Each recall**, from its cue to 50 after the reference's output falls silent: 2550–3140, 3550–4130 and 4350–4930.

That is 225 of the 586 moments (38%). The windows match where the reference's input or output changes by more than
0.1 between two samples, except that they also keep each recall peak's plateau. The comparison at each moment is
unchanged.

Fitness of the key designs, where fitness = (1 + output score) / 2:

| Design | Recalls | Whole trial | Relevant moments | Relevant + hold (1900–2550) |
|---|---|---|---|---|
| Reference | 12 | 1.000 | 1.000 | 1.000 |
| Memory field rebuilt | 12, in order | 0.968 | 0.967 | 0.972 |
| Release rebuilt | 12, in order | 0.959 | 0.955 | 0.965 |
| Skips D | 6 | **0.970** | 0.936 | 0.950 |
| From scratch | none | 0.871 | 0.779 | 0.799 |
| Silent | none | 0.761 | 0.761 | 0.763 |

How the restriction scores against the earlier checks:

- **Judged replays.** Every o and t replay now scores above the from-scratch design. It stays above 6 of the 8 g and h
  replays. Favourites ranked above the rest: AUC 0.71 → 0.78. o,t ranked above g,h: 0.92 → 0.98.
- **Unchanged checks.** Reference repeats 1.000. All 8 sweeps still peak at the reference's value. The path from
  silence to the reference still rises monotonically.
- **Timing.** Errors cost more. A whole output 30 late scores 0.86 instead of 0.95, and 100 late 0.70 instead of 0.87.

**Blind spot.** Whatever happens outside the windows is free. Each of these scores 1.000, against 0.98–0.99 over the
whole trial:

- a stray peak in a rest;
- a peak held after its event is shown;
- D recalled early, during the hold.

A B peak that is never released costs only through the 50-long release window (0.995).

Also scoring every moment at which the design's own output has a peak closes the blind spot, but it is rejected. It
lifts the from-scratch design to 0.834, because its held peaks sit on the reference's pre-shape bumps.

Scripts: `analyse_relevant.py`, run on `profiles/`, which now also holds the four run bests.

**Implemented** for the task, which now has 23 stages:

- the demonstration: "trigger", then "event D", "event C", "event A" and "event B", each 100 long;
- each of the three recalls: X; recall D, C, A and B; and a 50-long "release".

The fitness is the mean over those stages' moments.

- **The hold is left out.** The user first asked for the hold before the first recall to count, then for the times
  where nothing happens to be dropped.
- **Checked with `neat-dnfs-sol-eval`.** It gives the same scores as the analysis: reference 1.000, recalls in order
  0.967, release rebuilt 0.955, skips D 0.936, never recalls 0.779, silent 0.761.
- **Fixtures.** The from-scratch, skips-D and recalls-in-order designs are now test fixtures:
  `tests/fixtures/serial-order-profile-never-recalls.dnf`, `-skips-d.dnf` and `-recalls-in-order.dnf`.

## Literature (Undermind deep search, thesis workspace)

- **Concordance correlation:** Lin 1989.
- **Problems inherent to MSE- and NSE-based calibration:** Gupta et al. 2009.
- **MSE's failings as a fidelity measure:** Wang and Bovik 2009.
- **The double penalty of displaced features**, from spatial forecast verification: Roberts and Lean 2008; Ebert 2008.
- **Optimising dynamic neural fields to data and templates:** Igel, Erlhagen and Jancke 2001; Fix 2013. There is no PDF
  of either, so their error measures are unchecked.
