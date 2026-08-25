---
name: project-code-review
description: Review a diff against neat-dnfs's own standards - backwards compatibility, Clean Code, neat_tools/ helper reuse, C++20 idiom, evolution hot-path performance, and test coverage. Use before opening a PR or when asked to review a branch or change.
---

# Project code review

Review `git diff origin/main...HEAD` (or the diff you were given) against the checks below.
This deliberately overlaps `.coderabbit.yaml` so problems surface locally before CodeRabbit
posts them on the PR. That file's `instructions:` block is the only written code standard in
this repo - there is no `CONTRIBUTING.md` - so read it as the authority when in doubt.

## 1. Backwards compatibility (highest severity)

Not a finding to note - a **stop condition**. If the diff breaks compatibility, say so
first and stop; do not continue reviewing style.

Look for: changed public signatures under `include/`, altered default parameter values,
changed behaviour of an existing solution or genome operator, and any change to the genome /
population JSON serialization that older saved runs under `data/` or `templates/` would not
survive.

## 2. Clean Code

- Names that read like prose. `n`, `tmp`, `data2` are findings.
- Small, single-responsibility functions. A function that needs a comment to explain its
  sections wants to be several functions.
- **No raw owning pointers.** `shared_ptr` / `unique_ptr` only.
- No comments that explain *what* the code does. Only *why*, and only when non-obvious.
- Clarity over cleverness - if a reader has to stop and decode a line, it is a finding.

## 3. Helper reuse

Every new free function: does something equivalent already live in `neat_tools/`?

Search `include/neat_tools/utils.h`, `logger.h`, `config_loader.h`, `resource_paths.h`,
`solution_registry.h`, `ablation_presets.h`, and `include/constants.h` before accepting it.

Also flag helpers defined at a call site that *should* move into `neat_tools/` - a utility
function sitting in a solution `.cpp` is misplaced.

## 4. Modern C++20

Flag missed opportunities where they genuinely improve the code:

- `std::span` instead of `const std::vector<double>&` for read-only array views
- `std::string_view` for read-only string parameters
- `[[nodiscard]]` on accessors and anything whose result must not be dropped
- ranges, structured bindings, `std::format` over manual concatenation
- `enum class` over bare `enum`

Do not flag idiom for its own sake. If the change is cosmetic and the file is consistent
as-is, leave it.

## 5. Performance - the evolution loop

The hot path here is evolution, not a per-timestep simulation loop: `Population::evolve()`
runs generations over populations of genomes, each of which runs a full DNF simulation to be
evaluated.

- Copies of heavy objects - genomes, populations, species containers, field matrices, anything
  returned by value from an accessor
- `shared_ptr` copies (atomic refcount) inside per-generation or per-genome loops
- Allocation inside the generation loop that could be hoisted or reused
- Repeated map/string lookups per genome

**Thread safety is a live concern, not theoretical.** `PopulationParameters::parallelEvolution`
defaults to `true` and evaluation runs concurrently via `std::async`, so shared mutable state
touched during evaluation is a real bug, and the `sanitize` CI job (tsan) will find it. Flag
any new shared state reachable from a solution's evaluation.

## 6. Tests

- New or changed behaviour in `neat/`, `solutions/` or `neat_tools/` with no test
- **A new test file not added to `CMakeLists.txt`** - it is silently never compiled and the
  suite goes green having never run it. There is no `GLOB`. Check this explicitly every time.
- Floating-point assertions without an explicit tolerance - Catch2 wants
  `REQUIRE_THAT(x, Catch::Matchers::WithinAbs(expected, tol))`, not a bare `REQUIRE(a == b)`
- A test that would pass against the unfixed code, i.e. does not actually pin the bug
- A new evolution-tier test tagged `[Evolution]` without a seeded genome - an unseeded stub
  forms no field bump in a small fraction of runs, so the test flakes

## 7. Static analysis

`.clang-tidy` sets `WarningsAsErrors: '*'` with
`HeaderFilterRegex: 'include/neat/.*|include/constants\.h'`. A tidy diagnostic in those headers
does not warn, it **fails the Static Analysis job**. Treat a new finding there as a build break,
not a style nit.

## Output

Order findings most severe first. For each: file:line, one sentence on what is wrong, and
the concrete failure it causes or the standard it violates.

Say "clean" when it is clean. Do not manufacture findings to look thorough - a short honest
review beats a padded one.
