# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **CodeRabbit PR review** — `.coderabbit.yaml` with project-specific rules: C++20 idioms, no raw owning pointers, Catch2 test coverage enforcement, Doxygen completeness, and a welcoming tone for first-time contributors
- **Gemini issue triage** — `gemini-issue-triage.yml` auto-labels and posts a welcome comment on every new issue using the Gemini free tier; label creation and `gh` write commands are deterministic shell steps hardened against prompt injection
- **Gemini doc-sync check** — `gemini-doc-sync.yml` audits Doxygen, README, and CHANGELOG completeness on PRs touching `neat-dnfs/include/**`; skipped on forked PRs to avoid secret-missing failures
- **vcpkg maintenance** — `vcpkg-maintenance.yml` monthly cron creates a dependency version report issue using pure shell and `gh` CLI (no LLM required)
- **Release-baseline test coverage** — extends the Catch2 suite to establish a verified baseline before tagging a release (closes #20, #21, #23, #24, #25, #26):
  - `tests/test_solutions_tasks.cpp` (new) — construction, `clone()`/`copy()` independence, `mutate()`, and one real `evaluate()` per previously-untested task class (`MemoryInstability`, `SelectionInstability`, `MemoryTrace`, `InhibitionOfReturn`, `AND`, `XOR`); `DelayedMatchToSample` instead gets a regression test documenting a pre-existing bug (#47) that makes its `evaluate()` always throw
  - `tests/test_population_file_manager.cpp` (new) — exercises `PopulationFileManager`'s per-generation and end-of-run writes; cleans up its own output directory afterward
  - `tools::utils::normalize`, `normalizeWithGaussian`, `normalizeWithFlatheadGaussian` — boundary, peak/decay, and flat-top behaviour
  - `Solution::hasTheSameTopology`/`hasTheSameParameters`/`hasTheSameGenome`, `clearGenome`, `translatePhenotypeToGenome`, and `crossover()` edge cases (identical parents, fitter-parent inheritance)
  - `Population::setSize`/`setNumGenerations`, `getSolutions`/`getSpeciesList`, and the `stop`/`pause`/`resume`/`start` control surface
  - `Species::copyChampionToNextGeneration`, `randomlyAssignRepresentative`, `hasFitnessImprovedOverTheLastGenerations`, `isExtinct`, and single-member `crossover()`
  - `Genome::removeConnectionGene`, `isEmpty`, `clearLastMutations`, and excess/disjoint/difference metrics at identical and fully-disjoint boundaries
  - `FieldGene`/`ConnectionGene` `clone()` independence, `clearLastMutations()`, and `FieldGene::setAsHidden()`
  - `test_helpers.h::resetGlobalState()` — resets the three process-global statics (`Genome::globalInnovationNumber`, `Solution::uniqueIdentifierCounter`, `Species::currentSpeciesId`) that are not reset between `TEST_CASE`s, so the suite is order-independent under `--order rand`

### Changed
- `tools::utils` RNG — replaced per-call `std::random_device` + `std::mt19937` construction with a `thread_local` xoshiro256++ engine seeded once per thread, eliminating redundant reseeding overhead on every `generateRandomInt`/`Double`/`Float`/`Signal` call (~970x faster in microbenchmark) (closes #6)
- `Population::evaluate()` — replaced unbounded per-solution `std::async` fan-out (one OS thread per solution every generation) with a fixed worker pool sized to `hardware_concurrency()`, work-stealing over an atomic index; exceptions from any worker now propagate reliably instead of being dropped when an earlier future threw (closes #45)

### Fixed
- `tools::logger::log()` raced on a shared global `Logger` object when called concurrently from parallel solution evaluation, risking a message being emitted with another thread's level/colour; replaced with a per-call temporary and removed the now-unused shared global. `std::cout` writes in `log_cmd` are now serialised with a mutex (closes #5)
- `Genome::getInnovationNumberOfTupleWithinGeneration()` — removed this unused locked variant of the innovation-number lookup; it had zero callers and, being a plain `std::mutex`, would have self-deadlocked if ever called from inside an already-locked context such as `addConnectionGene()` (closes #3)
- `generateRandomSignal()` returned `-1` with probability 2/3 instead of the intended 50/50, biasing every mutation step direction in `FieldGene` and `ConnectionGene` mutation
- Per-thread RNG seed hardened against a degraded `std::random_device` (some implementations fall back to a deterministic sequence) by mixing in a per-thread hash and an incrementing counter, preventing threads from ever sharing a seed

---

## [1.1.0] - 2026-06-13

### Added
- **Catch2 unit test suite** — 70 tests / 4 213 assertions covering `Genome`, `Population`, `Species`, `FieldGene`, `ConnectionGene`, and all `Solution` subclasses (closes #8)
- **Codecov integration** — dedicated `coverage` CI job with GCC `--coverage` instrumentation, lcov report generation, and upload via `codecov/codecov-action@v4`; Codecov badge added to README (closes #9)
- **Doxygen documentation** — Doxyfile config, targeted API comments across public headers, and automated publishing to GitHub Pages on every push to `main` (closes #6)
- **Static analysis** — dedicated `static-analysis` CI workflow running clang-tidy and cppcheck; badges added to README (closes #7)
- **Cross-platform CI** — GitHub Actions workflow building on Ubuntu, Windows, and macOS with vcpkg; build scripts for each platform (`build.sh`, `build.bat`, `build_macos.sh`) (closes #5)

### Changed
- `Population`: removed `testMode` flag; test-specific behaviour now handled by proper test fixtures and solution stubs
- `Population::evolve()`: innovation-number map now guarded by a mutex to prevent data races during parallel evaluation
- Test suite expanded from partial Catch2 scaffolding to full coverage across all core components

### Fixed
- `localtime_s` portability — replaced with `#ifdef` guard selecting `localtime_r` on POSIX and `localtime_s` on MSVC (closes #3)
- `Population::evolve()` crash when called in test context — root cause was missing `initialize()` call and absent testMode guard; resolved via proper test fixture setup
- MSVC `C2665` error in tests — `ElementDimensions` push-back now uses explicit brace-initialisation
- Transitive include breakage in `examples/incremental_evolution.cpp` and `examples/solution_evaluation.cpp` — added explicit `#include <dnf_composer/simulation/simulation_file_manager.h>` after upstream dnf_composer header reorganisation
- lcov negative-counter error from parallel test execution — added `-fprofile-update=atomic` compiler flag and `--ignore-errors negative,gcov` to lcov commands
- OpenGL pre-find and `localtime_r` on non-Windows when building against imgui-platform-kit
- VS multi-config generator imported-target mapping for `MinSizeRel`/`RelWithDebInfo`

### Removed
- Stale empty test files and unused Catch2 scaffolding replaced by the new test suite

---

## [1.0.0] - 2024-01-01

### Added
- Core NEAT-DNF framework for evolving Dynamic Neural Field architectures
- `Genome` class encoding field genes and connection genes with NEAT-style innovation numbers
- `Population` class managing the full evolution loop: evaluation, speciation, reproduction, selection
- `Species` class for speciation and compatibility distance calculation
- `Solution` abstract base class with 20+ fitness helper methods for field-dynamics-based evaluation
- `FieldGene` supporting Gaussian and Mexican Hat kernel types with mutable parameters
- `ConnectionGene` representing inter-field kernel couplings with spatial parameters
- Parallel fitness evaluation via `std::async`
- Elitism and pruning-based selection
- Per-generation statistics: fitness, species diversity, architectural complexity
- JSON serialisation of solutions via `nlohmann-json` and `SimulationFileManager`
- Implemented task hierarchy:
  - **Detection Instability** — transient input-driven activation
  - **Memory Instability** — self-sustained activation without input
  - **Selection Instability** — winner-take-all competition
  - **Delayed Match-to-Sample** — internal memory biasing later selection
  - **Inhibition of Return** — delayed inhibitory bias against previously selected locations
  - **AND** and **XOR** — logical tasks for validation
  - Additional tasks: `SingleBump`, `SelfSustainedSingleBump`, `MemoryTrace`, `SelectTheObject`, `TwoRobotTeam`, `TimingResponse`, `SelectiveOutputField`
- Three runnable executables: tabula rasa evolution, incremental evolution from template, solution evaluation
- Pre-evolved solution templates in `templates/` (JSON format)
- Post-hoc analysis tools in `analysis/` (Python/Streamlit visualiser)
- Configurable hyperparameters via `include/constants.h`
- Catch2-based test infrastructure (partially populated)
- Logger with console and ImGui output modes, per-level colour coding

### Dependencies
- C++20, CMake 3.31.6+, VCPKG
- [`dynamic-neural-field-composer`](https://github.com/Jgocunha/dynamic-neural-field-composer)
- [`imgui-platform-kit`](https://github.com/Jgocunha/imgui-platform-kit)
- `imgui`, `implot`, `imgui-node-editor`, `nlohmann-json` (via VCPKG)

[Unreleased]: https://github.com/Jgocunha/neat-dnfs/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/Jgocunha/neat-dnfs/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Jgocunha/neat-dnfs/releases/tag/v1.0.0
