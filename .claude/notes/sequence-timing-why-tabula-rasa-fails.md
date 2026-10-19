# Sequence timing: why tabula rasa evolution fails, and what to change

Assessment as of 18 October 2026, after from-scratch runs 12-33 on the boost variants of `sequence-timing`. Run data
for runs 23-33 is in `sequence-timing-tabula-rasa-runs-23-33.md`. Evolution has solved the task only from seeds that
already contain most of the architecture (seed b2: run 18, 0.983; seed b3: run 10, 0.990). From scratch the best is
0.740 (run 29), and no run has built per-event timing.

## Why it fails

### Nothing pays until the whole timing mechanism exists

Every working solution times each event the same way:
- a memory field holds the events
- a slow integrator ramps at each event's position
- a clamp freezes the ramps between demonstration and recall
- X releases the clamp, and the output fires as each ramp crosses threshold

A partial chain scores no better than none, so fitness gives no direction toward it.

Evidence: run 18's solution, evolved from seed b2, needs 6 fields and 9 of its 10 connections; removing any one breaks
it. Runs 24-33 grew 6-8 fields on average, but none built this chain.

### Cheaper behaviours sit next to it, and each is a local optimum

Each time one shortcut is closed, evolution moves to the next cheapest:
- the burst, with all events firing together right after X (0.60-0.70)
- fixed-lag replay, timed from the demonstration rather than X (0.81, run 23)
- grouped order, with events firing in two or three simultaneous groups (0.72-0.74)

Moving from any of them toward per-event timing first breaks what already scores, so fitness drops.

Evidence: run 23 reached fixed-lag replay. Under the probe gate, 6 of 10 runs stayed in the burst and 4 settled on
groups. All plateaued within 50-80 generations.

### The integrator is a needle in a haystack

The slow integrator needs a tau around 1e4-1e5 and an input amplitude around 300-1000, both at once. A new field
arrives with a random tau, random kernels and random connections, at a rate of 1% per genome per generation. Getting
both values right together by chance is rare.

Evidence: scaling the hand-built integrator's tau and input together by 0.3 still works (0.976), but by 0.1 drops to
0.899 and by 0.03 to 0.746.

### Switching peaks off is a second all-or-nothing piece

Ending each recall peak needs its own inhibitory structure: a past-events field (hand-built design), erasing each memory
as its event is recalled (run 18), or inhibiting the output directly (run 20). Without it, peaks pile up, and f6 is
stuck at about 0.42 for every pile-up solution.

Evidence: f6 is 0.37-0.45 in most from-scratch runs, and in run 23 it stayed at 0.42 from generation 58 to 199.

### Speciation protects innovation poorly in both regimes

- At compatibility threshold 4.0, new structure merges into old species and is outcompeted before it is tuned.
- At 1.5, the population of 1000 splits into hundreds of species of a few genomes each, which weakens selection
  within each species.
- Stagnation culling sends all of a stagnant species' offspring to the single best species. With many small species
  stagnating at once, diversity periodically collapses.

Evidence: run 12 at 4.0 kept 2 fields and one species. Runs 24-33 at 1.5 ended with 166-665 species, and runs 23 and
27 collapsed to a single species.

### A weighted sum hides partial progress

A mutation that improves order but costs output silence or isolation can net zero or less, so steps toward timing are
often invisible to selection. This is a hypothesis; the runs so far did not test it directly.

### Confounds that are now removed

Runs 12-15 ran with the elitism bug (fixed in `ceb3c16`): elitism carried the all-time high-water genome, which under
noisy evaluation may be a lucky one, and that genome was then mutated. Seeds with gaps in their field ids also hit the
hidden-field id bug (fixed in `f0d4493`). Runs 23-33 are free of both.

## What to change, in the order to try

1. **A curriculum in which each stage needs exactly one new mechanism.** Each stage rewards behaviour, never the
   existence of structure (rewarding hidden fields is ruled out), and is seeded with the previous stage's champion
   through `neat-dnfs-inc-evol`.
   - Stage 1: recall which events occurred, in any order (memory).
   - Stage 2: one event at the right delay after X (integrator and clamp).
   - Stage 3: two events, in order and on time.
   - Stage 4: the full four-event sequence, with peaks switching off.

   Runs 16 and 19 tried a two-stage version (order only, then the full task) and failed. The jump between their stages
   needed several mechanisms at once.
2. **Fewer events in early stages.** One timed interval needs a far smaller architecture than four events and three
   intervals.
3. **Make the integrator reachable.** Draw a new field's tau from a few time scales (fast about 10, medium about 100,
   slow about 1e5) instead of one log-uniform range, so integrator-like fields appear often. A slow field also needs
   strong input, so the two settings would have to be designed together. (A per-target-tau cap on connection
   amplitude was proposed on 16 October and rejected; this is about where new fields start, not a cap.)
4. **Protect innovation better.**
   - Adjust the compatibility threshold during a run to hold a target number of species (for example 20-50), instead
     of a fixed 1.5 or 4.0.
   - Give a new species a grace period before stagnation culling.
5. **Reward behavioural diversity.** Add a novelty score on the output's onset pattern, so that leaving the burst for a
   new pattern is rewarded even before it scores well.
6. **Multi-objective selection.** Rank solutions by Pareto front on the partial terms instead of a weighted sum, so
   gains in order or timing are not hidden by losses elsewhere.
7. **More attempts per idea.** Use cheaper early-stage evaluations, with fewer trials and shorter recalls, to afford
   more independent runs per setting. Ten runs of 100 generations took about 18 hours.

Recommendation: start with the curriculum (1 and 2), since it addresses the first two causes directly. Add the tau
time scales (3) next, and the speciation changes (4) if stages still stall.

## Evaluation loopholes found so far, and their fixes

Useful when a new run reports a suspiciously high score: check its output in dnf-composer before trusting it.

| Loophole | Found in | Fix |
|---|---|---|
| Always-active output sweeping past the events | run 6 (fixture `sequence-timing-wave-exploit.dnf`) | Recall credit scaled by silence before the cue |
| Replay tuned to a fixed set of gaps | level-2 runs | Gaps drawn afresh from each third of 300-1500 |
| Output held hundreds below rest, so it passed as silent and its peaks were tiny | run 20 | f7, output near rest (`d16e95f`) |
| Fixed-lag replay from the demonstration, firing without X | run 23 | Probe gate on f4-f6 (`0cf8fb5`) |
| Chaotic sweep: many broad peaks all over the output for a few hundred after each X, crossing the event positions by chance; best-of-population selection keeps its lucky evaluations (reported 0.81, re-evaluated 0.67) | serial-order run 2 | Serial-order only: each cycle's f4-f6 scaled by output activity at the event positions, beyond chance (`7ea01c6`) |
| A peak rising while the previous one is still up, which goes within 50: isolation was only checked 50 after each onset, so it earned full one-at-a-time credit | a variant of the serial-order hand-built design, built to test this (fixture `serial-order-overlapping-peaks.dnf`) | Serial-order only: isolation also taken on the step each peak rises (`51d03d9`) |
