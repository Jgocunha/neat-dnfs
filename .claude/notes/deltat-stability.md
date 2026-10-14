# How far deltaT can be raised without changing behaviour

`Solution` builds every phenotype with `SimulationConstants::deltaT` (config), including phenotypes loaded
from a template, so one saved solution can be re-run at any step size. dnf-composer integrates with
forward Euler and scales `NormalNoise` by `amplitude / sqrt(deltaT)`, so noise stays consistent.

Measured on `sequence-timing` (2026-10-14): 100 saved solutions from a run evolved at deltaT = 1, evaluated
at deltaT 1, 1 (repeat - the noise floor), 2, 5 and 10. Drift = largest change in any partial fitness
against deltaT = 1. 49 of the 100 had non-trivial recall activity; the rest barely do anything.

| deltaT | time per eval | active solutions with drift > 0.05 | worst drift |
|---|---|---|---|
| 1 (repeat) | 1x | 0 / 49 | 0.001 |
| 2 | 0.5x | 0 / 49 | 0.008 |
| 5 | 0.25x | 3 / 49 | 0.375 |
| 10 | 0.15x | 24 / 49 | 0.982 |

- **deltaT = 2 is indistinguishable from 1** at half the cost, even with tau down to 5 (deltaT/tau = 0.4).
- **deltaT = 5 is mostly fine.** The three solutions that changed were not the lowest-tau ones (min tau 5, 15
  and 39). They sit near a bifurcation, where a slightly different threshold-crossing time flips the outcome.
- **deltaT = 10 breaks behaviour**, mostly for solutions with tau < 20 (37 of 50 in that band drift).

Task timings must be written in simulation time and converted to steps (`SequenceTiming::stepsFor`), or
raising deltaT silently stretches the protocol. Solutions evolved at one deltaT are tuned to it; re-check
them before comparing runs made at different step sizes.
