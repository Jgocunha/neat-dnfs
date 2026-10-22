# Serial-order profile: comparing the output's behaviour, with the hold (22 October 2026)

The first run from scratch under `profile-output-activation` (OA1) ended on a shortcut. Its output holds the events
above the field in reverse order (B +11 > A +8 > C 0 = D 0) and all four burst after each X. Late in each recall the
reference also holds the events still to come above its field, so that rule scored the shortcut above recalling D alone
(0.777 vs 0.756). The user asked for the activation to be encoded as behaviour, and for the hold before the first recall
to be scored.

## The rule

`SolutionConstants.profileComparesOutputBehaviour`, preset `profile-output-behaviour`, off by default. At each moment it
reads the output field `nf 2` at the shown events' positions (D 180, C 120, A 20, B 80) and compares three things with
the reference's:

- **Peaks.** Each event's output σ(u), and σ of the highest point farther than 15 from every event. They are compared
  as an overlap: 1 − Σ|design − reference| / max(Σ max(design, reference), 1). That gives 1 when neither peaks, 0 when
  they peak in different places, and a peak where the reference has none costs as much as a missing one.
- **Lifts.** Each event's activation minus the field's level (the median of the output). A design gets full credit
  once it is as far as the reference or further in the same direction, half at the field's level, none when as far the
  other way. Below one unit, the reference's lift is judged on a scale of one.
- **Order.** The six pairwise differences between the lifts, judged the same way. The pairs' shortfalls are summed in
  activation units over twice the reference's, so a pair counts by its size in the reference: a peak 30 above a trough
  outweighs two troughs 2 apart.

The moment's score is the mean of the three (two with one event shown). The hold before the first recall, from the end
of the last event's stage to the first recall's X (1900–2550), is always scored as one more stage, `hold`. Outside the
stages, with `profileCountsOutputActivityOutsideStages`, only the peaks are compared, each moment counting as much as
the output is active. The loader rejects the key together with `profileComparesOutputActivationOnly` or
`profileAlsoComparesFieldOutputs`.

The older rules do not score the hold. Adding it to them would change their results, so it comes only with this key.

## What the reference's behaviour is

| Moment | Peaks | Above the field | Below the field | Order |
|---|---|---|---|---|
| Trigger | – | – | – | all level |
| Demonstration, after A | – | D, C, A | – | D > C > A |
| Hold | – | D, C, A, B | – | D > C > A > B |
| Recall D | D | C, A, B | – | D > C > A > B |
| Recall C | C | A, B | D | C > A > B > D |
| Recall A | A | B | D, C | A > B > D > C |
| Recall B | B | – | D, C, A | B > D > C > A |
| After release | – | – | all four | D > B ≈ C > A |

Its bumps rise above the field no farther than 9 positions from their event, so the 15 that separate events from
"away from them" leave a margin.

## Scores

Offline mirror on recorded profiles; the C++ evaluation gives the same to 0.001.

| Design | Current rule (`profile-output-activation`) | This rule |
|---|---|---|
| Reference | 1.000 | 1.000 |
| L2 + C, A, B peaks and troughs (edit) | 0.899 | 0.923 |
| recalls-in-order (fixture) | 0.933 | 0.914 |
| skips-d (fixture) | 0.873 | 0.764 |
| L2's best: D alone on every cue | 0.756 | 0.706 |
| L2 with D erased after its recall (fixture) | 0.770 | 0.710 |
| C1: item memory, no recall | 0.671 | 0.583 |
| OA1's best: reversed order + bursts (fixture) | **0.777** | **0.575** |
| Silent output | 0.524 | 0.526 |
| o1, o4, o5, t6 (old-rule favourites) | 0.59–0.64 | 0.35–0.41 |
| Held peaks (never-recalls, A1, the 0.854 design) | 0.56–0.60 | 0.25–0.34 |

Steps added to L2's recorded output: C, A and B peaks +0.124 (+0.068 before), every trough +0.094 (+0.099), a real
trough at D +0.036 (+0.038). Erasing D to the field level earns +0.004 (+0.013). Half a trough is half a step, and it
loses again on the order of the troughs, where the reference keeps D shallowest.

All eight parameter sweeps around the reference peak at the reference's value, and the path from silence to the
reference rises steadily.

## What was tried and dropped

- **A margin on the activation.** Differences within ±m count nothing, everywhere. It compresses the scale toward 1,
  and the shortcut stays above D alone at every m tried (1, 2, 3, 5).
- **Comparing the change over time** (u(t) − u(t − lag)) as well. The shortcut falls only 0.004 below D alone at a lag
  of 50, and stays 0.008 above it at 100.
- **Scoring the hold with the activation.** The shortcut falls below D alone (0.761 vs 0.792), but item memory stays
  below it (0.726).
- **The field's own rise in the code.** It is meant for the trigger, but it costs memory designs whose field drifts
  differently from the reference's: item memory fell to silence's level.
- **Peaks event by event** (1 − |difference| for each) instead of the overlap. Silence earns three in four at every
  recall moment and rises to 0.74, compressing the scale; the shortcut's margin below D alone falls from 0.129 to
  0.053.
- **The order unweighted** (each pair's ramp averaged). Then the fine order of the troughs counts as much as a peak
  over a trough.
- **The older σ comparison outside the stages.** It gives a held peak where the reference is quiet about 0.5, not 0,
  which lifts held peaks to 0.48–0.51 and the favourites to 0.56–0.61. The overlap keeps one language for peaks inside
  and outside the stages.

## First run from scratch (OB1)

`data/Serial Order Profile Output Behaviour/2026-10-22 11h57m25s`, pop 1000, 200 generations, 23 minutes, frozen at
0f055ea. The best (solution 196162) scores 0.758 on all three recordings. It builds the ordered item memory, with no
peak during the demonstration and D > C > A > B above the field through the hold (1.00 on both). On every cue it
recalls D and releases it on time (recall D 0.98), through its output inhibiting its slow memory field at D. It is never
active outside the stages, and it is not OA1's shortcut. It stops at D alone, a cleaner one than L2's best (0.706):
nothing is pushed below the field, and C, A and B never peak (recall C, A and B 0.34, release 0.56).

Going on needs D pushed below the field and a drive that stays on after D's release, so that C, about 8 below threshold
then, crosses it. Its saved names top out at a lucky 0.785 (solution 198860, which records 0.74): pick a run's best by
re-recording it, not by the fitness in a file name.

## Known costs

- A peak at the wrong time costs more than no peak. Designs that hold peaks through the demonstration, the old
  favourites among them, score below silence.
- Item memory alone and OA1's shortcut tie (0.583 vs 0.575); only recalling D (+0.12) separates them.
- Real inhibition must push D below the field to pay. Erasing it earns almost nothing, as OA2 showed under the previous
  rule: a run can still stall at D alone.

Scripts: `.claude/temp/output-activation-runs/behaviour_code*.py` (prototype and variants), `behaviour_calibrate.py`
(fixture values), `behaviour_outside_check.py` (outside check), `.claude/temp/output-behaviour-runs/summarise.py` (run
summary, same rule as the C++).
