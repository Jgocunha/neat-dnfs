---
name: work-an-issue
description: End-to-end workflow for resolving one GitHub issue in neat-dnfs - branch, TDD, build, review, docs, PR. Use when asked to fix, implement, or close a specific issue number.
---

# Work an issue

One issue, one branch, one PR. Do not stack unrelated changes.

## 1. Understand

```bash
gh issue view <N> --comments
```

Restate in your own words: the goal, and the **success criterion** - the observable thing
that will be true when this is done. If you cannot state a success criterion, the issue is
too vague to work unattended; stop and say so.

## 2. Branch

Never commit to `main`. Branch from an up-to-date base:

```bash
git fetch origin
git switch -c <type>/<slug> origin/main
```

Branch type is one of `bug/ feat/ chore/ ci/ docs/ test/ refactor/`. Keep the slug short
and specific: `bug/prune-count`, not `bug/fix-the-pruning-problem`.

## 3. Failing test first

Write the Catch2 test that reproduces the bug or specifies the feature. **Add the file to
`CMakeLists.txt`** if it is new - there is no `GLOB`, so an unlisted file is never compiled and
your "passing" test never ran. Build and **watch it fail** - a test that passes before you have
implemented anything is testing nothing.

Follow the suite's existing conventions: `REQUIRE_THAT(x, Catch::Matchers::WithinAbs(v, tol))`
for floating point, the helpers in `tests/test_helpers.h` and `tests/test_stub_solution.h`
rather than rolling your own genome, and `seedFixedGenes()` on any stub that needs a field bump
to form reliably.

Keep it out of the `[Evolution]` tag unless it genuinely needs a full evolution run - that tag
is the slow, stochastic lane.

## 4. Instrument before theorising

If the failing test does not immediately tell you *why*, *do not reason about the control
flow by reading source*. Add logging, run it, and read what actually happened:

```cpp
tools::logger::log(tools::logger::DEBUG,
    "genome 12 gen 7: best fitness 0.41, 3 species, champion unchanged for 4 generations");
```

Write messages as plain natural-language sentences carrying the values that matter. Guessing
at the flow costs more than instrumenting it. Remove the instrumentation before the PR unless
it earns a permanent place at `DEBUG`.

## 5. Implement

Minimally. The smallest change that makes the test pass and nothing more.

Before writing any helper function, **search `neat_tools/` first** - `utils.h`, `logger.h`,
`config_loader.h`, `resource_paths.h`, `solution_registry.h`, `ablation_presets.h`, and
`include/constants.h`. If something close already exists, extend or reuse it. A new free
function duplicating an existing helper will be rejected in review.

If the change is to a tunable, check whether `config/` JSON overrides it at runtime before
editing `constants.h` - behaviour here is JSON-driven, and editing the constant alone can be a
silent no-op.

## 6. Build and test

Use the `build-and-test` skill. Green means the full suite, not just your new test - and say
which lane you ran. `-LE slow` passing is not the same claim as the whole suite passing.

## 7. Review your own diff

Use the `project-code-review` skill on `git diff origin/main...HEAD`. Fix what it finds
before asking anyone else to look.

## 8. Docs

Use the `docs-check` skill. Doxygen on new public API (CI enforces it), the matching prose or
config JSON, and no stale references left behind.

## 9. Ship

```bash
git add -A && git commit -m "<type>: <lowercase summary>"
git push -u origin <branch>
gh pr create --title "<type>: <summary>" --body "<filled PR template>

Closes #<N>"
```

The body fills `.github/PULL_REQUEST_TEMPLATE.md` - use the `pr` skill to draft it.

Do **not** merge. The PR waits for human review.

## 10. Hand back

Report: PR URL, what changed, what you verified (with the test count and the lane), and
anything you deliberately left out.

## Stop conditions

Stop and ask rather than guessing when:

- **Backwards compatibility cannot be preserved.** This is never your call to make alone.
- The issue admits two readings that would produce materially different work.
- Tests fail for reasons unrelated to your change - report the pre-existing failure, do not
  paper over it or "fix" it as a side quest. In the `[Evolution]` lane, re-run and check
  against unmodified `main` before concluding anything.
- The fix requires a design or product decision the issue does not settle.

When you stop, say exactly what you completed, what is blocked, and what you need.
