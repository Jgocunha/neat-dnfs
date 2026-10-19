# Replaying a saved genome in dnf-composer

A champion's `.dnf` (under a run's `best_solutions/`) holds the network, not the protocol. To watch it do the task in
dnf-composer, add the task's stimuli as timed Gaussian stimuli. Four details decide whether the replay matches what
the evaluator scored. Each was found on 19 October 2026 by comparing replays with the evaluator's own trace.

1. **Zero the evaluator's leftover stimuli.** A saved genome keeps the evaluator's `gs nf 1 <position>` stimulus
   elements, wired into the input field, as they were at the moment of saving. Genomes from the Gaussian-cue
   `sequence-timing` task (runs 1-9) kept X's stimulus (`gs nf 1 50`) at amplitude 15. In a replay that holds X on
   permanently, and solved runs looked broken. Set every `gauss stimulus` element's amplitude to 0.
2. **Make the boost exactly uniform.** The evaluator's X boost is a `BoostStimulus`: 15 everywhere. A Gaussian 1000
   wide on the 200-wide field still varies by about 0.5%, and some genomes amplify that unevenness. Run 17's
   champion erupted over the whole field in the replay, but stayed silent in the evaluator. Use width 1e6 and
   `circular: false`.
3. **Step the stimuli before the fields.** dnf-composer steps elements in list order. A timed stimulus listed after
   the input field reaches it one step late. Put the timed stimuli first in the element list.
4. **End each window at onset + 50, not onset + 48.** The evaluator's pulses drive the field for 25 steps at deltaT 2.
   A window ending at onset + 48 gives 24. Designs that time events by integrating a pulse shift by up to 80 with one
   step missing. The old Gaussian hand-built design recalled 50-80 early and fired D during the gap.

With all four, the replays of solved runs (5 L3, 7, 9, 10, 18, 20) and of most failed ones match the evaluator's
onsets within a few time units at similar gaps. Two kinds still differ:

- **Noise-driven genomes** (sequence-timing runs 12, 25, 29, 31 and 33): the evaluator's own trials disagree with each
  other, with each event either joining the burst or firing about 1700 later. One replay shows one outcome.
- **Genomes the evaluator rebuilds differently:**
  - Run 11 carries a stray self-connection (`gk cg 7-7`) from the hidden-field id bug, which the decoder folds into
    field 7.
  - Runs 21 and 22 differ at every gap, for a reason not yet found. It is not noise, element order or stimulus
    strength.

A replay is evidence only once its onsets have been checked against the evaluator's trace for the same genome.

The scripts are `.claude/temp/analysis/all-runs-replays/make_task_replay.py` (one replay) and `build_all.py` (every
run), both gitignored scratch.
