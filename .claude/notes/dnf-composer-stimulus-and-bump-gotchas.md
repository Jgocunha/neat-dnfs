# dnf-composer: stimulus width and bump detection gotchas

Two facts about dnf-composer that matter when designing a task's stimulus positions.

## GaussStimulus `width` is sigma, not full width

`GaussStimulusParameters::width` is passed straight to `tools::math::gauss` /
`circularGauss` as the Gaussian's standard deviation. A "width 30" stimulus spreads over
roughly ±60 samples, so events placed 20-40 apart merge into one blob in the input field.
Pick a sigma well under half the spacing between event positions.

## Bump detection is not circular

`NeuralField::updateBumps()` scans the activation linearly from index 0 to the end. On a
circular field, a peak centred at position 0 (or at the far edge) is reported as **two**
bumps - one starting at index 0 and one ending at the last index - each with a centroid
pulled away from the true centre. Every bump-counting fitness helper in `Solution` then
sees two bumps where there is one.

Keep task stimulus positions away from the field edges (at least a few sigma), or score
edge peaks with a helper that merges the split bump. `sequence-timing` moved its start
event from 0 to 50 for this reason.
