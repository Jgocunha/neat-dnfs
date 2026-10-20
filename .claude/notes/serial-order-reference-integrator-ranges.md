# Serial order reference: how far the integrator's parameters can move

Date: 20 October 2026. The reference is `tests/fixtures/serial-order-fitness-test-reference.dnf`, scored with
`neat-dnfs-sol-eval --task serial-order-fitness-test`, where it scores 0.988. One parameter set was changed at a time;
everything else stayed as in the fixture.

The integrator is "nf 5". In the design as it was built, it had tau 1e5, h +0.6 and self-kernel gk 0. M drove it
through `gk cg 3-5` at +110 and C through `gk cg 4-5` at -110. Those are the starting values for everything below.
Since 20 October 2026 the fixture uses tau 1e4 and a drive of ±13, which scores 0.988-0.989 and lies inside the
task's ranges. A test guards that.

## Tau and drive trade off

Tau stays far longer than the trial (5850), so the integrator's rate is set by drive / tau. Lowering tau alone
breaks it: 0.80 at 3e4, 0.75 at 1e4, 0.70 at 1e3. Scaling the drive of both connections with tau keeps it working:

| nf 5 tau | Drive (±) | Fitness |
|---|---|---|
| 1e5 | 110 | 0.988 |
| 3e4 | 33 | 0.989 |
| 2e4 | 22 | 0.988 |
| 1e4 | 11 / 12 / 13 / 14 / 20 | 0.963 / 0.988 / 0.989 / 0.988 / 0.955 |
| 5e3 | 6 / 8 / 10 / 11 / 12 / 15 | 0.86 / 0.955 / 0.988 / 0.986 / 0.986 / 0.88 |
| 2e3 | 4 / 6 / 8 / 10 / 13 | 0.87 / 0.91 / 0.962 / 0.950 / 0.89 |
| 1e3 | 8 / 12 / 17 | 0.85 / 0.86 / 0.88 |
| 500 | 15 | 0.67 |

- The design holds down to a tau of about 5e3.
- Below that, the leak over a trial costs the gradient.
- At each tau the working drive is a band roughly ±10% wide, about one ×1.3 amplitude mutation step.
- Away from the band fitness falls gradually, not off a cliff.

## Resting level is a narrow window

At tau 1e5, nf 5's resting level:

| h | Fitness |
|---|---|
| -2 | 0.73 |
| -0.5 | 0.76 |
| 0 | 0.81 |
| 0.25 | 0.93 |
| 0.4 | 0.986 |
| 0.5 | 0.988 |
| 0.6 | 0.988 |
| 0.75 | 0.988 |
| 0.85 | 0.978 |
| 1.0 | 0.82 |

- At tau 1e4 with drive 12: h 0 scores 0.89 and h 1 scores 0.81.
- The window is about 0.45 wide, close to one resting-level mutation step (0.5).
- 1.0, the configured ceiling where upward mutations clamp, lies outside the window.

## Self-kernel amplitude does not matter

nf 5's gk amplitude 0, 1 and 3 all score 0.986-0.989, at tau 1e5 and at tau 3e4 / drive 33. The design does not need a
field kernel amplitude of 0.

## Where a field's tau and resting level come from

- With `tauMutationFactor` set, tau mutates by ×1.3 or ÷1.3, and `tauStep` is unused.
- Tau is drawn log-uniformly in [tauMinVal, tauMaxVal], and the resting level uniformly in its range, only in two
  cases:
  - for a fresh genome's input and output;
  - when a field is re-randomised (`...NeuralFieldGenerateRandomParametersProbability`).
- A hidden field added by `Genome::addGene` is instead a copy of the field at the source of the connection it splits:
  same tau, resting level and kernel. Its incoming connection gets `GaussKernelConstants` width, amplitude and
  amplitudeGlobal, so those three keys are live even with `variableParameters` on.
- Each field mutation picks one of three branches: kernel, kernel type, or neural field.
- Per field per generation, the serial-order-fitness-test config of 20 October 2026 gives:
  - a ×1.3 tau step 0.54% of the time;
  - a re-draw 0.4% of the time;
  - a resting-level step 1.8% of the time.

  From tau 10, an integrator at 5e3 or more is 24 ×1.3 steps away. So in practice it comes only from a re-draw
  landing in both [5e3, 1e4] and the resting-level window, with probability 9% × 2.1% = 0.19% per re-draw.
- The task's config now has tauMaxVal 1e4 and gaussAmpMaxVal 50, and the fixture's nf 5 has tau 1e4 / drive 13. The
  largest connection magnitude in the reference is then 30 (`gk cg 7-2`).
- The same config puts connection sign flips at 0.05 and connection kernel re-draws at 0.02 of a connection's
  mutations, with the kernel branch at 0.93.
- Field kernel and field re-draws stay at 0.1 of their branch. They are the only route to Mexican hat fields and to
  slow or positive-resting fields.
- With those field rates, 29% of a reference-sized genome's offspring get at least one sign flip or re-draw. With
  connections at 0.2 and 0.1 it was 59%.
