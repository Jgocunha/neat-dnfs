# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

# Project: neat-dnfs

Everything above is general engineering guidance. Everything below is specific to this
repository and takes precedence where the two overlap.

## Layout

The git root holds `README.md`, `CHANGELOG.md`, `codecov.yml`, `.coderabbit.yaml`, `.github/`
and `.claude/`. The project itself is nested one level down:

```text
neat-dnfs/
|-- include/          public headers - Doxygen lives here
|   |-- neat/         connection_gene, field_gene, genome, population,
|   |                 population_file_manager, solution, species
|   |-- solutions/    task definitions - and, xor, dmts, ior, memory_trace,
|   |                 detection/memory/selection_instability
|   |-- neat_tools/   logger.h utils.h key_listener.h ablation_presets.h
|   |                 config_loader.h resource_paths.h solution_registry.h
|   +-- constants.h
|-- src/              mirrors include/
|-- tests/            Catch2 -> target neat-dnfs-test
|-- apps/             3 executables; apps/README.md documents the CLI
|-- config/           runtime hyperparameter JSON
|-- templates/        starting-solution JSONs
|-- analysis/         Streamlit visualizer + pytest suite
|-- scripts/          setup/build/package for windows, linux, macos
+-- docs/             Doxyfile
```

Everything is one namespace, `neat_dnfs` (plus `neat_dnfs::paths`).

## Build and test

Use the `build-and-test` skill. Two facts that bite:

- **`CMakeLists.txt` lists every test source explicitly.** There is no `GLOB` anywhere in this
  project. A new file under `tests/` that isn't added to that list is silently never compiled,
  and the suite goes green without ever having run it.
- **CTest has two lanes.** `ctest -LE slow` is the default lane - 196 tests, under ten seconds.
  `ctest -L slow` is the `[Evolution]` tier - it runs real `Population::evolve()` at
  populationSize=50 / numGenerations=10 / numRuns=5 per solution, so it takes tens of seconds
  and it is *stochastic* - re-run, and check against unmodified `main`, before reading a
  failure there as a regression.

There is no `CMakePresets.json`; the `scripts/` wrappers are the build path. `VCPKG_ROOT` must
be set or CMake hard-fails.

## Conventions

**Backwards compatibility is mandatory.** If a change cannot preserve it, stop and ask.
Never proceed on an assumption about what is safe to break.

**Clean Code always.** Deviating needs an explicit justification in the PR description.
Meaningful names, small focused functions, no comments explaining *what*, no raw owning
pointers (`shared_ptr` / `unique_ptr` only). The written standard is the `instructions:` block
in `.coderabbit.yaml` - there is no `CONTRIBUTING.md` in this repo.

**Check `neat_tools/` before writing any helper.** Utility, logging, config and path helpers
have existing homes and belong there, not beside their call site:

| Need | Goes in |
|---|---|
| General utility | `include/neat_tools/utils.h` |
| Logging | `include/neat_tools/logger.h` |
| Config / hyperparameter loading | `include/neat_tools/config_loader.h` |
| Resource / data paths | `include/neat_tools/resource_paths.h` |
| Solution registration | `include/neat_tools/solution_registry.h` |
| Ablation presets | `include/neat_tools/ablation_presets.h` |
| Tunable constants | `include/constants.h` |

Search these first. Re-implementing something that already exists in `neat_tools/` is a review
failure, not a style nit.

**Modern C++20** - ranges, concepts, structured bindings, `std::span`, `std::string_view`,
`[[nodiscard]]`.

**TDD** - write the failing test first, watch it fail, then implement. "Fix the bug" means
"write a test that reproduces it, then make it pass".

**Instrument, don't guess.** When tracing a bug, add logging and read the output rather than
reasoning about control flow from the source. Use
`neat_dnfs::tools::logger::log(logger::DEBUG, ...)` from `include/neat_tools/logger.h` -
levels `DEBUG/INFO/WARNING/ERROR/FATAL`, `Logger::setMinLogLevel()` for the threshold. Write
messages as plain natural-language sentences carrying the values that matter: *"genome 12 gen 7:
best fitness 0.41, 3 species, champion unchanged for 4 generations"* is worth far more than
`here1` or a bare variable dump - people and models both read them. Remove instrumentation
before the PR unless it earns a permanent place at `DEBUG`.

**Write findings down.** Something learned while working that is not obvious from the code
goes in a short, self-contained `.md` file - one topic per file, readable without the
conversation that produced it:

| Finding | Goes in |
|---|---|
| True of the project anywhere | `.claude/notes/` - tracked |
| Only true of this machine (toolchain versions, install locations, local paths) | `.claude/local-notes/` - gitignored |

**Temp files** go to **`<git root>/.claude/temp/`** - the `.claude/` beside `README.md`, *not*
one inside `neat-dnfs/`. Its contents are gitignored. If it does not exist, create it **with an
absolute path**: a relative `mkdir -p .claude/temp` run from the nested project root silently
creates a second, wrong `.claude/` one level down. That has already happened here - the stray
`neat-dnfs/.claude/temp/segv/` is what it looks like. Never scatter scratch files
through the project tree.

**Doxygen** - new or changed public entities under `include/` need `@brief`, one `@param` per
parameter, and `@return` unless void. This is enforced: `gemini-doc-sync.yml` runs a
completeness check on every PR touching `neat-dnfs/include/**`.

**Behaviour is JSON-driven, not compiled in.** Hyperparameters load at runtime, later wins:

```text
config/neat_dnfs.json -> config/solutions/<task>.json -> config/ablations/<name>.json -> CLI flags
```

`apps/README.md` documents the flags and worked examples. Changing a default usually means
editing JSON under `config/`, not a constant in `constants.h` - check which one actually
governs before editing either.

## Naming

Branches - `bug/`, `feat/`, `chore/`, `ci/`, `docs/`, `test/`, `refactor/` plus a short
slug: `bug/prune-count`, `feat/mechanism-ablations`.

Commits - conventional, lowercase, optional scope. Types in use across the history:
`fix, feat, test, perf, docs, refactor, ci, chore, style`.

```text
fix: clamp inhibitory amplitude to a negative range
test(speciation): cover extinct-species cleanup
chore: optimize CI workflows and scripts
```

Note the split: bugfix *branches* use `bug/`, their *commits* use `fix:`.

PRs - concise title, and the body fills `.github/PULL_REQUEST_TEMPLATE.md`.

## Release (maintainers)

Version lives in `neat-dnfs/CMakeLists.txt` (`NEAT_DNFS_VERSION_MAJOR` / `MINOR` / `PATCH`).
Bump per SemVer, add a `CHANGELOG.md` entry, commit `release: vX.Y.Z`, then tag and push -
`release.yml` triggers on the tag and **refuses a tag that disagrees with `NEAT_DNFS_VERSION`**.
