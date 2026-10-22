# Serial-order profile: comparing the output's activation alone (22 October 2026)

After the overnight runs the user set a new direction for `serial-order-profile`: forget the input field and score the
output's whole activation profile in the stages (the first boost, each demonstrated event, each recall), because the
negative activation of recalled events is what tells evolution that it needs a mechanism.

## The rule

`SolutionConstants.profileComparesOutputActivationOnly`, preset `profile-output-activation`, off by default:

- only the output field `nf 2` is compared, never the input;
- inside the 23 stages, its activation u, by the per-moment regularised concordance as before;
- outside them, with `profileCountsOutputActivityOutsideStages`, only its output σ(u), each moment counting as much as
  the output is active then. The loader rejects it together with `profileAlsoComparesFieldOutputs`.

Why each part:

- **The input adds nothing.** It is a fixed detection field (`fixedInputField`) that no connection can target, so it
  matches the reference's for every design (1.00) and only halves every difference.
- **σ inside the stages hid the troughs.** It is blind below threshold. Removing every trough from the reference cost
  0.015 under the overnight rule (`profile-full-outputs`) and costs 0.059 under this one.
- **u outside the stages would pay held peaks.** never-recalls holds peaks through the demonstration, where the
  reference holds its highest bumps below threshold: 0.557 on the stages alone, 0.669 with u outside, 0.555 with σ.

## What the reference does, against the user's three phases

| Phase | Reference output |
|---|---|
| First boost | Rises from −13 to −0.7, no peak |
| Demonstration | Pre-shapes D +11.3 > C +9.0 > A +7.6 > B +6.0 above the field: oldest highest |
| Recall | The current event peaks; the next ones wait below threshold, in order; the previous ones sit in troughs, deeper the more recent (D −6.7, C −8.9, A −10.1 below the field) |

The field stays at −13 through recall C, A and B and rises to −4 only from the release on, so the rise touches only the
three 50-long release stages.

## Scores under the rule

Offline mirror on recorded profiles; the C++ evaluation gives the same to 0.001.

| Design | Score |
|---|---|
| Reference | 1.000 |
| Reference without its troughs | 0.941 |
| recalls-in-order (fixture) | 0.933 |
| skips-d (fixture) | 0.873 |
| L2's best with every recalled event's trough laid on its own field (edit) | 0.855 |
| L2's best: recalls D alone on every cue | 0.756 |
| C2, D1, L1: D alone | 0.731–0.760 |
| C1: item memory, no recall | 0.671 |
| o1, o4, o5, t6 (old-rule favourites) | 0.59–0.64 |
| The 0.854 design: peaks held between stages | 0.596 |
| never-recalls (fixture) | 0.555 |
| Silent output | 0.524 |

No weighting of stages can rank o1, o4, o5 and t6 above the D-only designs. They score lower in every behaviour,
recall C, A and B included (0.50–0.61 against 0.53–0.63), because their peaks come at other times and their fields
rest at other levels.

## Weights by behaviour importance do not help

The user asked for the weights to follow behaviour importance. Counting each moment once already gives the
recall-related stages 78% of the score, the demonstration's memory 18% and the first boost 4%, close to the weights the
user chose for `serial-order-fitness-test` (0.75, 0.10 and the rest to constraints). Two alternatives, offline:

| Rule | D only − memory | Memory − worst shortcut | D's trough | All troughs | Memory − silent |
|---|---|---|---|---|---|
| Each moment once (adopted) | +0.059 | +0.075 | +0.038 | +0.099 | +0.147 |
| A share per stage kind (recall 0.5 and so on) | −0.010 to +0.016 | +0.04 to +0.07 | +0.031 to +0.041 | +0.085 to +0.116 | +0.08 to +0.12 |
| Positions weighted by the reference's features | +0.082 | +0.049 | +0.044 | +0.115 | +0.125 |

A trough is a small feature inside the 200-position field at every moment, so no stage weight singles it out. A fixed
share for the moments outside the stages is free credit to every quiet design. Position weights cross the "no explicit
peak weighting" rule and narrow the margin over shortcuts. The user chose to keep one weight per moment.

## Real inhibition is a small step

L2's best recalls D on the reference's timing (30–230 after X against 20–230), then lets D fall back to a bump 7.7
above the field where the reference holds it 6.7 below. Hand-built additions, recorded in dnf-composer
(`neat-dnfs-record-profile`):

- **One connection from L2's own field 7 back to the output**, inhibitory (a single mutation): field 7 is active only
  during D's peak, so it ends the peak early (130–170 after X) and leaves no trough. 0.725–0.734.
- **The reference's release loop added as a new field**: fed by the output, self-exciting, inhibiting the output at D,
  reset by X through the input. In about 540 variants none both holds a trough and lets X reset it before D's next
  recall. A strong hold makes troughs 4–12 below the field but loses D in cycle 2, where C wins instead. A reset
  that works leaves D at the field level, not below it.
- **The best stable step**, D pulled to the field after each recall (tau 55, rest −4.5, gain 12, inhibition 20, the
  reset kernel's global amplitude −0.8), scores 0.770 against L2's 0.756. Noise moves either by 0.001. The overnight
  rule scores the same pair 0.896 and 0.894.

The rule makes inhibition pay, which the overnight rule did not, but a real mechanism earns about +0.014 against the
+0.039 of an ideal trough at D. The two designs are the fixtures `serial-order-profile-d-only.dnf` and
`serial-order-profile-d-only-inhibited.dnf`.

## Stopping

`Population::endConditionMet()` stops a run when every partial fitness, here every stage, beats the target, not when
the fitness does. A design without troughs cannot stop a run, since its recall and release stages stay low.

Scripts and recorded profiles are in `.claude/temp/output-activation-check/` (gitignored).
