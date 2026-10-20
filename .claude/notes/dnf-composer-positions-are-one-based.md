# dnf-composer positions are one-based

dnf-composer samples field position (i + 1) * dx at index i of a field's components:

- a Gaussian stimulus at position p peaks at index p / dx - 1 (`tools::math::circularGauss` builds its x range with
  `std::iota(..., 1)`);
- the bumps a field reports use the same positions: `NeuralField::updateBumps` sets `startPosition = (i + 1) * dx`,
  so a bump from a stimulus at 180 has its centroid at 180.0;
- `TimedGaussStimulus::onTimes` are in simulation time `t`, not steps: at deltaT 2 a window of 50 is 25 steps.

`Solution::clampedIndexForPosition` mapped p to index p / dx until 20 October 2026, one sample to the right. So
`preShapednessAtPosition`, `negativePreShapednessAtPosition` and `negativePreShapingDepthAtPosition` read p + 1, and
a pre-shape or trough exactly at p scored as if it were one sample off. It now maps p to index p / dx - 1, and the
tests tagged `[FieldHelpers]` in `tests/test_solution.cpp` check that every activation-reading helper reads the
position it is given. The fix shifts the AND, DMTS and HRI packaging scores slightly, since each scores a pre-shape
or trough with one of those helpers.

To build an activation profile in a test where dnf-composer would put it, use `dnf_composer::tools::math::circularGauss`
itself (`addGaussianProfile` in `tests/test_solution.cpp`), rather than computing indices by hand.
