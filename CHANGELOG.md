# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **`sequence-timing` task** — a preliminary task inspired by Wojtak et al. (2023), *Adaptive timing in a dynamic field architecture for natural human–robot interactions*. One input and one output field, 200 wide. A start cue X and four events D, C, A, B are demonstrated as transient pulses at intervals of 100, 500, 200 and 1000 time units; later only X is given, and the output must peak at D, C, A and B in turn, at the demonstrated intervals and one at a time. Each evaluation runs three recall trials, with gaps between demonstration and cue drawn afresh on every evaluation (one from each third of 300-1500), so that only timing measured from X can succeed and no fixed set of gaps can be learned, plus a probe trial without X in which the output must stay silent. Seven partial fitness terms cover input representation, input return to rest, output silence outside recall, recalled events (rising after the cue), recalled onsets measured from X (tolerance proportional to the onset), one-peak-at-a-time, with the last peak dying out on its own, and the output staying near its resting level (full credit while its lowest activation averages within 60 below rest, outside recall and during it, weight 0.05), so couplings that hold the output hundreds below rest no longer pass as silence. Each recall trial's credit is scaled by how silent the output stayed before its cue, so an always-active output cannot earn recall credit from peaks that sweep past the events, and every recall credit term is also scaled by how silent the output stays in the probe trial, so a replay at a fixed lag after the demonstration, which also fires without X, cannot earn recall credit in the trials where X happens to come first
- **`sequence-timing-boost` task** — `sequence-timing` with the start event X delivered as a homogeneous boost of the input field instead of a Gaussian at position 50, so X carries no position and can only act through couplings that respond to the input field's total activity. Same events, trials and fitness terms; input representation scores the event pulses only
- **`sequence-timing-boost-order` task** — `sequence-timing-boost` whose recalled-timing term is half order credit and half exact onsets, a stepping stone for evolving the boost task from scratch. Each consecutive pair of recalled events earns the time between their onsets as a fraction of 100 time units, capped at 1, so an output that ignites at every event at once earns almost nothing for order and each step towards a sequence earns more. A solution that recalls the sequence at the demonstrated times scores as in `sequence-timing-boost`. Every `sequence-timing` variant's evaluation trace now also records each recall's order credit
- **`sequence-timing-boost-order-only` task** — `sequence-timing-boost` whose recalled-timing term is order credit alone, the first stage of a curriculum whose champions seed `sequence-timing-boost`. In `sequence-timing-boost-order`, exact timing pulls the first event out of an early burst of simultaneous peaks, which reverses the first pair and cancels the order credit; without the exact-onset half, every peak that rises after its predecessor earns credit wherever it lands
- **`serial-order` task** — serial order on its own, without timing. The demonstration gives X as a boost of the input field, then D, C, A, B as pulses, all evenly spaced 200 apart, since only their order is recalled; the short demonstration also cuts each evaluation's simulated time by about a sixth against the `sequence-timing` intervals. After a gap drawn from 300-1500, X is given three times, each starting a recall cycle whose length is drawn afresh from 800-1500; after every X the output must peak at D, C, A and B in turn, each alone, at its own pace, and fall silent before the cycle ends. Each peak must be the only one in the output on the step it rises, so the previous one must have gone first: a peak that only stands alone once it has settled no longer earns full one-at-a-time credit. A probe trial without X must stay silent. Same seven partial terms as `sequence-timing`, with order credit as the timing term and the recall terms averaged over the three cycles, so a sequence recalled on the first trigger only earns a third of the recall credit. A hand-built competitive-queuing design (primacy gradient in a slow integrator, a done field cleared by every X), wired only with connections evolution can make, scores 0.994
- `Solution::addBoostStimulus()` / `removeBoostStimuli()`: add a homogeneous boost of a field, and remove every boost, mirroring the Gaussian-stimulus helpers
- `SolutionParameters::evaluationTrace`: a task can record what its last evaluation observed as a JSON document, and the structured per-generation overview (`overview.jsonl`) saves it under `bestSolution.evaluationTrace` when present. `sequence-timing` records, per trial, when each demonstrated pulse started, output silence in each phase and each recalled event's onset and isolation (after settling, and on the step its peak rose), plus every output peak that rose after the cue. Tasks that record nothing write the same overview as before
- `Solution::getNeuralFieldOrThrow()` is now `protected` instead of `private`, so a task can read a field's bumps step by step during its own evaluation
- Optional `ConnectionGeneConstants.gaussAmpMinVal` / `gaussAmpMaxVal` / `gaussAmpMutationFactor` give Gauss connection kernels their own amplitude range, drawn log-uniformly and mutated multiplicatively when the factor is set, and `ampGlobalStep` their own global-amplitude step. When absent, connections keep sharing the field kernels' range and steps
- Optional `ConnectionGeneConstants.ampGlobalMinVal` / `ampGlobalMaxVal` bound a connection kernel's global amplitude separately from a field's own kernel, so a task can let connections carry uniform excitation without giving fields runaway self-excitation. When absent, connections keep sharing the field kernels' bounds, so existing configs behave as before
- Optional `AblationConstants.allowOutputAsConnectionSource` and the `output-feedback` preset that sets it: mutation and connection seeding may also start a connection at an output field (never output → itself), which they otherwise never do. Off when absent, so existing configs load and evolve as before; the preset writes to `data/<Task> Output Feedback/`.
- Loading a template (`--template` in `neat-dnfs-inc-evol` and `neat-dnfs-sol-eval`) logs a warning for each connection evolution could not have made under the current rule, such as one out of an output field: offspring inherit it, but no run from scratch can build it. `Genome::connectionTuplesEvolutionCannotMake()` lists them.
- Optional `NeuralFieldConstants.tauMutationFactor`: when set, a field's tau is drawn log-uniformly in `[tauMinVal, tauMaxVal]` and mutates by multiplying or dividing by the factor, so one range can span time scales orders of magnitude apart. When absent, tau keeps its uniform draw and additive `tauStep`

### Fixed
- Decoding a phenotype whose connection names carry innovation numbers (`gk cg 1-3 7`) now reserves those numbers, so connections added by later mutations are never given them again. Before, an incremental-evolution run seeded from a saved solution could give two different connections the same innovation number, corrupting compatibility distances and crossover alignment
- Elitism now carries each generation's best solution into the next one, unchanged. It used to re-insert the genome holding the all-time fitness high-water mark, which under noisy evaluation (the DNF simulation's noise, or tasks that draw their trials afresh on every evaluation) can come from one lucky evaluation of a genome that is no longer the best; that genome was then mutated in place, while the actual current best was lost whenever its species had five members or fewer. The elitism validation check now verifies that the previous generation's best survived
- A new hidden field gene now takes an id one above the largest existing id, instead of "field count + 1". A genome whose field ids have gaps - a template that skips ids, such as `nf 1, 2, 3, 5, 7`, or a crossover offspring - used to get a duplicate id, so the field could not be built ("already exists in phenotype") and its connections duplicated existing ones. Genomes with contiguous ids get the same ids as before

---

## [0.3.0] - 2026-09-01

### Added
- **Structured per-generation run output** — when `PopulationConstants::saveStructuredOverview` is enabled (the default), a run writes `overview.jsonl` (line-delimited JSON, one object per generation) alongside the existing prose `per_generation_overview.txt`, whose format is unchanged (closes #82):
  - named fields for everything the prose line already carried, so the analysis tooling no longer has to reverse-engineer a 12-group regex
  - the full per-generation fitness distribution (min/max/mean/median/stddev/q1/q3), not just average and best
  - per-species membership sizes, and the best solution's lineage as a structured `parentIds` pair rather than a fragment of `Solution::toString()` prose
  - gated by a new `PopulationConstants::saveStructuredOverview` flag (default `true`), following the existing `saveXxx` pattern
- **Evolution-loop profiling** — a new `NEAT_DNFS_PROFILE` CMake cache option (mirroring `NEAT_DNFS_SANITIZER`, default `OFF`) gates a header-only RAII scoped timer in `neat_tools/profiler.h` (closes #80):
  - times the four `evolve()` phases (`evaluate`, `speciate`, `upkeep`, `reproduceAndSelect`) plus the per-generation file-I/O path
  - writes `profile.csv` into the run directory, one row per generation, over a fixed column set so rows stay positionally stable even when a phase is not exercised
  - compiles to nothing when the option is off: `ScopedTimer` becomes an empty type and the query functions fold to their zero/empty answers
  - only the main thread records; `evaluate()`'s parallel worker path is deliberately left uninstrumented so no cross-thread accumulation is needed (TSan-safe by construction)
  - `upkeep` is *inclusive* of `save` (the save timer nests inside it), so columns do not sum to the generation total — documented on `profiler::snapshot()`
  - first measurement on the `and` task at population 8: file I/O costs roughly as much per generation as evaluation itself, confirming the bottleneck this issue suspected
- **Golden tests for phenotype build and genotype round-trip** — a new `[Golden]` CTest lane (`ctest -L Golden`) pins down *what value* `buildPhenotype()` and `translatePhenotypeToGenome()` produce, rather than only that they run without throwing (addresses #75; tiers 3-4 remain blocked on #44):
  - tier 1 diffs each template's built phenotype against a checked-in fixture under `tests/golden/`
  - tier 2 asserts the genotype->phenotype->genotype round-trip is lossless — field-gene count, id, type, connection-gene count and innovation numbers are all preserved, for every task, unconditionally
  - `detection-instability`, `memory-instability` and `selection-instability` declare more fields in their topology than their template supplies, so the remainder are constructed with **unseeded random** kernel parameters; tier 1 skips them with a warning rather than checking in a fixture that would fail on its next run, until seeding lands (#44)

### Fixed
- **The genotype<->phenotype round-trip is now lossless for every task**, closing #64 and #47:
  - `translatePhenotypeToGenome()` inferred each field gene's role (INPUT/OUTPUT/HIDDEN) from its neural field's phenotype connection degree — a heuristic that cannot work, since `FieldGene` builds all three roles with identical element topology. The role is now recovered from the gene's id via `initialTopology`'s declared ordering, which the decode already parsed and previously discarded
  - every `ConnectionGene` constructor built its coupling kernel at the global `DimensionConstants::xSize` regardless of the fields it actually joins; a mismatched coupling silently fails to connect (`dnf_composer` logs and returns) and is then lost on the next decode. This was the real cause of `dmts` losing all three of its connection genes — its template was never malformed, its couplings were just being rebuilt at the wrong width. Constructors now take the dimensions to build at, defaulted to the previous global
  - connection-gene innovation numbers were renumbered from scratch on every decode instead of being recovered from the coupling kernel's own element name (`"gk cg <in>-<out> <innov>"`), which every connection this codebase writes — and every saved champion under `data/` — already carries
  - `dmts`'s fields represent hue and are tuned for a 360-wide field; `DimensionConstants::xSize` was lowered 360 → 100 for `InhibitionOfReturn` (commit `fe58ccbbf`) without retuning this task, so `DelayedMatchToSample::evaluate()` threw unconditionally. `config/solutions/dmts.json` now overrides `xSize` back to 360 for this task alone

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
