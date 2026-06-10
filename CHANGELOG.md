# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Cross-platform CI/CD workflow via GitHub Actions
- Automated release workflow with pre-built binaries per platform
- Codecov integration for test coverage reporting
- GTest unit test suite covering all core NEAT-DNF components
- Automated static analysis with clang-tidy
- Doxygen documentation published to GitHub Pages
- Refactor `examples/` → `experiments/` with per-experiment executables and JSON config files
- Deprecate compile-time `include/constants.h` in favour of runtime config loading

### Bug Fixes (pending)
- Fix `localtime_s` portability issue in `logger.cpp` (MSVC-only call, breaks Linux/macOS)
- Fix RNG performance regression — `mt19937` engine recreated on every call
- Remove dead `finalMessage` assignment in `Logger::log()`
- Investigate and document thread-safety contract for global innovation number map
- Enable population integrity validations in Debug builds
- Remove large commented-out code block from `logger.cpp`
- Uncomment and fix disabled test files in `CMakeLists.txt`

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

[Unreleased]: https://github.com/Jgocunha/neat-dnfs/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Jgocunha/neat-dnfs/releases/tag/v1.0.0
