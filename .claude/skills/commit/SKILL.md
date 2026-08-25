---
name: commit
description: Read the uncommitted diff in neat-dnfs and print conventional commit message(s) for it, grouped by logical change. Use when asked to write, suggest or draft a commit message, or "what should I commit this as".
---

# Commit

Reads the working tree and prints commit messages to chat. **Never runs `git add` or
`git commit`** - staging and committing stay a human decision.

## 1. Read the diff

```bash
git status --short
git diff HEAD
```

Include untracked files - `git status --short` marks them `??`; read their contents too,
they are not in `git diff HEAD` by default.

If the tree is clean, say so and stop. Do not invent a message for nothing.

## 2. Group by logical change, not by file

A diff touching three unrelated things gets three messages, not one. Group by concern:
a genome fix here, a new test there, a CI tweak elsewhere - even if they landed in the same
session. Do not force unrelated work into a single commit to keep the list short.

## 3. Write each message

Per `.claude/CLAUDE.md` > Naming - use what the project actually uses, not a generic
convention:

- Types in use: `fix, feat, test, perf, docs, refactor, ci, chore, style`, plus `release:`
  for version bumps.
- Lowercase, imperative mood, no trailing period.
- Optional scope: `type(scope): summary`.
- Bugfix *branches* use `bug/`, but their *commits* use `fix:` - never `bug:`.

```text
fix: clamp inhibitory amplitude to a negative range
test(speciation): cover extinct-species cleanup
chore: optimize CI workflows and scripts
```

## Output

A markdown bullet list, one message per logical change, each in a code span. When there is
more than one, name the files it covers so they can be staged separately:

```text
- `feat: add .claude skills and project CLAUDE.md` - .claude/CLAUDE.md, .claude/skills/
- `chore: track .claude CLAUDE.md, skills and notes` - .gitignore
```

Nothing else - no staging, no committing, no pushing.
