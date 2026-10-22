# Evolving serial order from scratch: overnight experiments, 21–22 October 2026

Brief from the supervisor: "You are the experimenter... The goal is to evolve from scratch." Task:
`serial-order-profile`, scored by how closely the input and output activation match the Gauss-only reference design's.
Everything below is from 19 runs between 22:21 and about 07:30, all at population 1000 unless stated. Run folders are
in `neat-dnfs/data/Serial Order Profile*/`; each has `config/README.md` and config snapshots. The working log is
`.claude/temp/overnight/LOG.md`.

## Bottom line

**Not solved.** From scratch, evolution now gets this far:

- a memory of each event below threshold, with a pre-shape in the right order, though shallower than the reference's;
- recall of the first event, D, alone and on time, on every cue;
- no stray peaks anywhere.

Four independent from-scratch lineages reach this point:

- two through the curriculum (C1 → C2, R1 → R2);
- one with no curriculum (D1);
- one continued from a full-task run (B1 → E1).

None goes on to C. Recall C, A and B, and the release, were not found under any setting tried:

- the full task;
- the recall-two step, with one species, with species at threshold 1.5 and with species at threshold 6;
- the one-event task, from three lineages;
- a 400-generation run;
- a population of 2000.

**Why it stops at D.** Going on to C needs two things at once:

- D's release, a field that holds "D was recalled" and is reset at each cue;
- a recall drive that lasts past the cue and lifts the output field.

Either alone costs fitness. I added the reference's release loop by hand to evolved designs to check this:

- **Without a reset at each cue,** it suppresses D for the rest of the trial.
- **With a reset,** it creates the trough after every recall but still scores lower than no release. That is because
  the reference's released state is a raised field with D below it, and the raised field needs the lasting drive.
- **The two halves exist, in different lineages.** The no-curriculum lineage (D1, O3) raises its field after recall
  as the reference does, but has no trough. The curriculum lineages have neither.

**The design change that mattered most** was comparing each field's output, its sigmoid, as well as its activation.
Every from-scratch run before it exploited the same blind spot: in the activation alone, a peak and a high subthreshold
bump in the same place look almost alike. Held peaks therefore scored like memory, and a missing recall peak cost
almost nothing.

## Design choices (all committed locally, not pushed; fast lane 370/370)

| Commit | Change | Why |
|---|---|---|
| ca00f9d | Score only the moments when something happens (approved before the night) | Static stretches let a never-recalling design score 0.871 |
| 3882608 | `profileCountsOutputActivityOutsideStages`: moments outside the stages count, weighted by the output's activity | Both runs on ca00f9d peaked where nothing was scored |
| 3882608 | `profileScoredStages`: score only part of the trial (curriculum) | To build the behaviour in steps |
| fa30a3f | `profileAlsoComparesFieldOutputs`: compare each field's sigmoid output as well as its activation | Held peaks scored like memory; missing recall peaks cost little |
| 737e439 | Finer curriculum steps: `recall-two`, `recall-three` | To isolate the step from D to C |
| 4176242, 5e1fe52, 0ee52c9 | Presets that let species form in large genomes | Every run after the first step held one species |
| 50d0a5e | `profileEventCount`: a curriculum over the sequence's length, with the reference recorded on 1- and 2-event schedules | On one event, release is worth twice as much |

Each option is off by default, so `serial-order-profile` without a preset scores exactly as at ca00f9d. The presets
are in `config/ablations/profile-*.json` and are documented in `apps/README.md`.

**This deviates from your "plain u(t)".** The field-output comparison adds a second view of the same profile. It is
your call whether that stays.

## Results

![Output field through the trial: the reference and four evolved designs. Shaded: scored stages; dots: output above
threshold.](serial-order-from-scratch-overnight-2026-10-22.png)

Folders are under `neat-dnfs/data/Serial Order Profile*/`. Each score is on that run's own curriculum level and
preset, so scores are comparable only within a level.

| Run | Setup | Best | What the best does |
|---|---|---|---|
| A1 | From scratch, stage 1 (demonstration), activation only | 0.922 | Holds peaks at D and C through the demonstration. The shortcut. |
| B1 | From scratch, full task, activity term | 0.893 | Pre-shapes, holds a weak D peak, recalls D on the first cue only, then flashes. |
| A2 | A1 → stage 2, activation only | 0.918 | Re-ignites the held D and C together at every cue. |
| **C1** | **From scratch, stage 1 with outputs** | **0.972** | **Holds a subthreshold bump for each event; no peaks at all.** |
| **C2** | **C1 → stage 2 with outputs** | **0.965** | **Recalls D alone, on time, on every cue; pre-shape D > C > A = B.** |
| C3 | C2 → full task with outputs | 0.892 | Stopped at generation 78: stalled. |
| E1 | B1 → full task with outputs | 0.892 | Recalls D on every cue, then nothing. |
| **R1** | **Replicate of C1** | **0.965** | **Target reached at generation 119: item memory, no peaks.** |
| **R2** | **R1 → stage 2 with outputs** | **0.965** | **Recalls D alone on every cue; pre-shape D > C > A > B, though shallow.** |
| C4 / R3 | C2 / R2 → recall-two | 0.938 / 0.935 | No progress. One species. |
| C5 | C2 → recall-two, species allowed (cutoff 100) | 0.937 | Stopped at generation 59: 646 species, so selection was close to random. |
| C6 | C2 → recall-two, threshold 6 over cutoff 100 | 0.938 | No progress, with up to about 30 species. |
| **D1** | **From scratch, full task with outputs, no curriculum** | **0.893** | **Recalls D alone on every cue; still climbing at generation 200.** |
| O1 / O2 / O3 | C2 / R2 / D1 → one-event task | 0.950 / 0.958 / 0.954 | Tuned the recall of D; no release. O3, from the no-curriculum lineage, raises its field after recall as the reference does (to −4.4 by cycle 3), but D's bump only shrinks and never becomes a trough. |
| L1 | D1 → full task, 400 generations | 0.898 | 0.893 → 0.898 in 400 generations (20–45 species). Still recalls D alone, then nothing. |
| L2 | C2 → full task, population 2000 | 0.893 at generation 128 | 0.889 → 0.893, one species. The same plateau. Still running at 07:15; it ends near 08:00 and its README follows then. |

**What reproduces.** From scratch, comparing field outputs as well:

- **Item memory.** The demonstration stage gives subthreshold item memory 2 out of 2 times (C1, R1).
- **Recall of D.** The next stage gives recall of D alone on every cue 2 out of 2 times (C2, R2).
- **Without a curriculum,** the full task reaches the same point in about 200 generations (D1).
- **Activation only.** Without the output comparison, every from-scratch run ends on held peaks or flashes (A1, B1,
  and the earlier runs 18h18m17s and 21h04m14s).

## The step from D to C, measured

**What each change would be worth.** I edited C2's recorded output in each recall-C window and scored the result at
the recall-two level.

| Edit | Score |
|---|---|
| None | 0.934 |
| D released into a trough | 0.941 |
| C peaking with D not released | 0.963 |
| Both | 0.969 |
| D's peak held on (a lasting drive, no release) | 0.927 |

C2's D peak ends only because its drive fades with the cue. So C cannot win unless D is held down and the drive
lasts, and the drive alone holds D up, which costs.

**Release by hand** (`.claude/temp/overnight/release_reachability.py`). I added the reference's release loop to O1's
best and scored it on the one-event task (base 0.933):

| Loop | Effect | Score |
|---|---|---|
| Self-exciting, without the reset at each cue | The trough appears after the first recall, then D is never recalled again | 0.81–0.88 |
| Without self-excitation | No trough | 0.88–0.95 |
| Self-exciting, reset by X | Correct one-event behaviour on every cue: recall, trough, recall | 0.918 |

**Behaviour is not what scores.** The last row behaves correctly yet still scores lower than not releasing. The
reference's released state is a field raised to −4 with D 6.7 below it, and the raised field comes from its recall
drive lasting after the cue. Profile matching rewards reproducing this particular architecture's way of releasing, not
release as such.

**Speciation and task length do not cross the valley.**

| Setting | Result |
|---|---|
| One species | No progress |
| About 30 species | No progress |
| About 650 species | Drift |
| One-event task, where release is worth twice as much | No release in 3 lineages |
| 400 generations (L1) | No progress |
| Population 2000 (L2) | No progress |

## Next steps, for your decision

1. **Keep the field-output comparison and the activity term.** With them, from-scratch runs stop ending on held peaks
   and flashes, and reach memory and the first recall reliably. Without them, every run found a shortcut. This is the
   one deviation from "plain u(t)", so it is your call.
2. **Decide what counts as success for release.** A design can release correctly (recall, trough, recall again) and
   still score below one that does not release, because it does not raise its field the way the reference does. Two
   options:
   - accept that from scratch must also discover this reference's raised field;
   - score the post-recall state more loosely.
3. **Make the release structure reachable by mutation, not by reward.** A working release is a field that is fed by
   the output, inhibits it locally, holds itself, and is reset by the cue. Every partial version is penalised. Two
   mutation operators could help, both outside the fitness (per "reward behaviour only"):
   - a structural mutation that adds a field with both connections at once (output → new field, new field → output),
     as NEAT's add-node adds one in-and-out pair;
   - new hidden fields that start with random time scales and self-excitation, instead of copying their source field
     (tau 10, rest −10).
4. **Optional curriculum stage: score the rests on the one-event task.** Offline this rewards the full post-recall
   state strongly (0.887 → 0.986), but it also brings back the copy-the-rests shortcut (0.957). I did not run it.
