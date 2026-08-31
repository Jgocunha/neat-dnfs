# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **Structured per-generation run output** — when `PopulationConstants::saveStructuredOverview` is enabled (the default), a run writes `overview.jsonl` (line-delimited JSON, one object per generation) alongside the existing prose `per_generation_overview.txt`, whose format is unchanged (closes #82):
  - named fields for everything the prose line already carried, so the analysis tooling no longer has to reverse-engineer a 12-group regex
  - the full per-generation fitness distribution (min/max/mean/median/stddev/q1/q3), not just average and best
  - per-species membership sizes, and the best solution's lineage as a structured `parentIds` pair rather than a fragment of `Solution::toString()` prose
  - gated by a new `PopulationConstants::saveStructuredOverview` flag (default `true`), following the existing `saveXxx` pattern

---

## [0.2.0] - 2026-08-26

### Added
- **Run provenance metadata** — every run directory under `data/<TaskName>/<timestamp>/` gains a `run_metadata.json` alongside `evolution_timestamps.txt` when overview persistence is enabled (`PopulationConstants::saveOverview`, on by default), so a run's fitness or performance can be attributed to its environment rather than guessed at (closes #73):
  - `neat_tools/build_info.h.in` (new, `configure_file`-generated) — compiler ID/version, CMake version, `NEAT_DNFS_VERSION`, `NEAT_DNFS_SANITIZER`, build type, git SHA and dirty flag, the resolved commit of `imgui-platform-kit`/`dynamic-neural-field-composer`, and `vcpkg list` output, all captured at CMake configure time
  - `neat_tools/machine_info.h`/`.cpp` (new) — OS name, logical core count, CPU model, and total RAM, gathered at process start (`#ifdef _WIN32`/`__APPLE__`/`__linux__` behind a portable interface)
  - `PopulationFileManager::saveRunMetadata()` — writes the `build`, `dependencies`, `machine`, and `run_parameters` sections; gated by the same `PopulationConstants::saveOverview` flag as the timestamps file
  - Seed and resolved-config fields are intentionally omitted for now — they depend on GitHub #44 (seeding) and the config-resolution work, neither of which has landed yet
  - **Analysis dashboard** — a new "Provenance" page (`analysis/viz/views.py::render_provenance_view`) shows one selected run's recorded build, dependencies, machine, and run-parameter facts, with an amber badge on a dirty build; backwards compatible with every pre-existing run that predates this file, showing a plain "not recorded" state instead of an error at the file, section, and field level

---

## [0.1.0] - 2026-08-24

First published release. Nothing before this was ever tagged or released, so the
two sections below are development history rather than releases anyone could
install; they are numbered 0.0.x to keep this file monotonic.

### Added
- **Pre-built binaries for Windows, Linux and macOS** — `.github/workflows/release.yml` builds, packages and publishes ready-to-run archives when a `v*` tag is pushed, so an experiment can be run without VCPKG or a compiler (closes #13):
  - `neat_tools/resource_paths.h`/`.cpp` (new) — `paths::resourceRoot()` locates `config/` and `templates/` at runtime instead of through the compile-time `PROJECT_DIR` macro, trying `$NEAT_DNFS_ROOT`, the executable's own directory, `<executable>/../share/neat-dnfs`, then the source tree the binary was configured from. A build tree still resolves to `PROJECT_DIR`, so development and `ctest` are unchanged; a downloaded archive resolves to itself, which it previously could not do at all
  - `paths::dataRoot()` — a relocated binary writes its `data/` results under the working directory rather than into the source tree it was built from; `$NEAT_DNFS_DATA_DIR` overrides it
  - `install()` rules split into a `dev` component (library, headers, package config — unchanged) and a `runtime` component (the three executables, `config/`, `templates/`, and on Windows the vcpkg DLLs), so a release archive carries only what running an experiment needs
  - `scripts/package.sh` / `scripts/package.bat` (new) — build one platform's archive from an existing Release build tree, named `neat-dnfs-<version>-<platform>-<arch>`
  - The workflow smoke-tests each archive on a clean path, from a directory that is neither the build nor the source tree, before publishing it; release notes come from this file's section for the tag
  - `tests/test_resource_paths.cpp` (new) — candidate selection order, the error naming every path tried, and the source-tree fallback a build tree relies on
- **Versioning** — `NEAT_DNFS_VERSION` in `CMakeLists.txt` gains a patch component and is now the single source of truth for release archive names, set to `0.1.0` (was `1.0`); the release workflow refuses to publish a tag that disagrees with it. Pre-1.0 is deliberate: the API is still free to move
- **Mechanism ablation studies** — `AblationConstants`/`AblationPresets` (new `include/neat/ablation_presets.h`, `src/neat/ablation_presets.cpp`) gate five NEAT mechanisms independently, selectable at runtime via `neat-dnfs-evol --task NAME --ablation NAME` with zero source edits (closes #76):
  - `no-growth-io-only` / `no-growth-one-hidden` — freeze structural mutations, seeding every legal connection (and one hidden field, respectively) at `initialize()` instead
  - `no-speciation` — `Species::isCompatible()` short-circuits to always-compatible; paired with a random non-minimal initial population per Stanley & Miikkulainen (2002, §5.5)
  - `no-crossover` — `Species::crossover()` reproduces by mutation only, cloning the single parent
  - `random-initial-topology` — seeds 1-5 random hidden fields and a random legal-connection subset instead of the minimal input/output genome
  - `Genome::legalConnectionTuples()`/`seedAllLegalConnections()`/`seedRandomConnections()` extract the connection-legality rule shared by the mutation path and the new seeding path, so they cannot drift apart
  - Each ablated run's data folder is suffixed with the ablation name (`PopulationFileManager::setFileDirectory()`), so ablated and control runs list as separate experiments in the analysis dashboard with no extra setup
  - `apps/` collapses from ten task-specific binaries to three (`neat-dnfs-evol`, `neat-dnfs-inc-evol`, `neat-dnfs-sol-eval`), each taking `--task`/`--ablation`/`--runs`/`--pop`/`--gens`/`--target`/`--template`/`--list` via a small hand-rolled parser (`apps/solution_registry.h/.cpp`)
  - `tests/test_ablations.cpp` (new) — baseline parity, per-preset seeding shape, structure-freeze across 1000 mutations, seeder/mutator rule agreement, clone-only offspring under no-crossover (with an enabled-crossover control), and `isCompatible()` short-circuit
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

## [0.0.2] - 2026-06-13

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

## [0.0.1] - 2024-04-19

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

[Unreleased]: https://github.com/Jgocunha/neat-dnfs/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Jgocunha/neat-dnfs/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Jgocunha/neat-dnfs/releases/tag/v0.1.0
