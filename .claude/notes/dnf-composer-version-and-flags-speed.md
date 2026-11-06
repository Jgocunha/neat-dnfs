# dnf-composer version and compiler flags do not change evaluation speed

Measured 30 Oct 2026. Neither a newer dnf-composer nor stronger compiler flags make a neat-dnfs evaluation meaningfully
faster, so a rebuild is not a way to shorten a run.

## Version

v2.12.1 (9 Oct 2026) changes nothing in the stepping path against v2.11.0: no diff in `src/elements/`, `tools/math` or
the SIMD dispatch. Its changes are `[[nodiscard]]`, the `.dnf` format version and load order, UI, docs, tests and `/W4`.
The large speedups in the dnf-composer changelog landed earlier. The hybrid FFT path is 2D only and neat-dnfs uses 1D.

neat-dnfs compiles unchanged against v2.12.1, and the default test lane passed 369/369. The `[Evolution]` lane was not
run against it.

## Flags

The library builds with `/O2 /Ob2`, no `/GL`, no `/fp:fast`. `/arch:AVX2` applies to one translation unit,
`tools/simd_dispatch_avx2.cpp`; the other 68 stay at the SSE2 baseline. `TRADE_OFF_CAVEATS.md` in
dynamic-field-theory-software saying the library is compiled with `/arch:AVX2` is out of date.

This costs little because the convolution and the sigmoid already run hand-written AVX2+FMA code, chosen by cpuid at
runtime. Those are the expensive parts of a step: Gauss kernels take 50-60% of it, fields 22-26%, noise 17-24%.

## Measurement

One-thread `neat-dnfs-sol-eval`, 3 rounds of 4 evaluations each, on a P core and an E core, for three designs: 162 runs
in 18 cells. Only the dnf-composer library differed between the builds. Each run read process CPU time on a pinned
logical CPU, with the variant order rotated, while two evolution runs saturated the machine. Speed is against the median
of three builds with identical code and flags. The ratios may differ on an idle machine.

| Variant | All | P core | E core | Faster in |
|---|---|---|---|---|
| v2.12.1 | x0.995 | x0.994 | x0.999 | 5/18 |
| `/arch:AVX2` | x0.979 | x0.961 | x1.008 | 7/18 |
| + `/GL /LTCG` | x0.981 | x0.961 | x1.012 | 5/18 |
| `/arch:AVX2 /fp:fast` | x1.006 | x0.992 | x1.037 | 11/18 |
| all three | x1.007 | x1.001 | x1.039 | 13/18 |

The three identical builds differ by 1.9% in a typical cell and 8.6% at most. Everything above is inside or close to
that band, apart from the E-core gain of about 4% with `/fp:fast`, and the P-core loss of about 4% with `/arch:AVX2`
alone, which appeared in all nine P cells and whose cause was not looked into.

Single runs came in up to x1.28 faster. They coincide with a low wall-to-CPU ratio, so the pinned core was less
contended for that run, and a control build did the same at x1.07. Treat them as load noise.

No variant changed behaviour. Field activations stayed within about 1e-9 of the installed library over 12000 steps, and
mean fitness matched within its spread on all three designs. See `dnf-composer-input-order-rounding.md` for the noise
floor.
