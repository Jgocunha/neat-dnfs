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

## Literature (Undermind deep search, thesis workspace)

- **Concordance correlation:** Lin 1989.
- **Problems inherent to MSE- and NSE-based calibration:** Gupta et al. 2009.
- **MSE's failings as a fidelity measure:** Wang and Bovik 2009.
- **The double penalty of displaced features**, from spatial forecast verification: Roberts and Lean 2008; Ebert 2008.
- **Optimising dynamic neural fields to data and templates:** Igel, Erlhagen and Jancke 2001; Fix 2013. There is no PDF
  of either, so their error measures are unchecked.
