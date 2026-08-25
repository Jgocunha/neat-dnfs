---
name: docs-check
description: Verify documentation keeps up with a code change in neat-dnfs - Doxygen on public API, the matching prose doc or config JSON, and stale references elsewhere. Use before opening a PR, or when asked whether docs need updating.
---

# Docs check

Three passes over `git diff origin/main...HEAD`. Report findings; do not silently rewrite
prose that is merely different from how you would put it.

## Pass 1 - Doxygen

Scope: new or changed public entities under `neat-dnfs/include/`.

Each needs:
- `@brief` - one line, what it does
- one `@param` per parameter, names matching the signature exactly
- `@return` unless the return type is `void`

Flag three failure modes:
- **Missing** - new public function/class/struct with no block.
- **Stale** - a parameter was renamed, added or removed and the block still describes the
  old signature. This is the most common and the easiest to miss.
- **Wrong** - the description contradicts what the code now does.

Private helpers in `src/` do not need Doxygen.

**This pass is enforced in CI.** `.github/workflows/gemini-doc-sync.yml` runs a completeness
check on every PR touching `neat-dnfs/include/**`, and the `docs` job publishes Doxygen to
GitHub Pages. A gap here becomes a PR comment whether or not you catch it first.

## Pass 2 - Prose and config

There is no local `wiki/` directory in this repo - narrative documentation lives in the GitHub
wiki, off-repo. What *is* in-repo and can go stale:

| Changed | Owes an update to |
|---|---|
| CLI flags, app behaviour, config precedence | `neat-dnfs/apps/README.md` |
| A hyperparameter default | the JSON under `neat-dnfs/config/`, not just `constants.h` |
| A solution's parameters or fitness | `neat-dnfs/config/solutions/<task>.json` |
| An ablation's meaning | `neat-dnfs/config/ablations/<name>.json` |
| Starting topology shipped with the project | `neat-dnfs/templates/` |
| Build, dependencies, setup | root `README.md` |
| Anything user-visible | `CHANGELOG.md` (Keep a Changelog, at the git root) |
| Visualizer behaviour | `neat-dnfs/analysis/` - no README, so check the module docstrings |

**Remember that behaviour is JSON-driven.** A change to a default is only real if the JSON
under `config/` changed too; editing a constant that the config overrides at runtime is a
no-op the tests may not catch. Check which one actually governs.

Rule of thumb: **if a user could notice the change without reading the source, something owes
them an explanation.** Pure internal refactors do not.

If the change is user-visible and the GitHub wiki covers it, say so and name the page - you
cannot edit it from here, so it becomes a note for the PR description rather than a fix.

## Pass 3 - Stale references

Grep the repo for references the change invalidated:

```bash
grep -rn "<old_symbol>" README.md CHANGELOG.md .coderabbit.yaml \
    neat-dnfs/apps/README.md neat-dnfs/config/ neat-dnfs/templates/ .claude/
```

Look for renamed or removed symbols, moved file paths, changed build commands or flags, JSON
keys that no longer load, and example snippets that no longer compile. Check the root
`README.md` and `.coderabbit.yaml` specifically when the build process or directory layout
moved - `.coderabbit.yaml` names paths and the `neat-dnfs-test` target explicitly and will
quietly go stale.

## Output

For each pass: either "clean" or a concrete list of file + line + what is wrong. If nothing
in the diff touches public API or user-visible behaviour, say so and stop - a refactor with
no API change legitimately needs no doc update.
