---
name: pr
description: Read the branch diff in neat-dnfs and print a PR title and description in chat, filling the repo's pull request template. Use when asked to write, draft or suggest a PR title/description, or "what should the PR say".
---

# PR

Reads the branch against `origin/main` and prints a PR title and body to chat. **Never runs
`gh pr create` or `git push`** - opening the PR stays a human decision.

## 1. Read the branch

```bash
git branch --show-current
git log origin/main..HEAD --oneline
git diff origin/main...HEAD --stat
```

If the branch is `main`, or there are no commits ahead of `origin/main`, say so and stop.

## 2. Title

Same conventional format as a commit - `<type>: <lowercase summary>`. If the branch's
commits share one type, use it; if they are mixed, pick the dominant concern rather than
concatenating types.

## 3. Body - fill the existing template

Use `.github/PULL_REQUEST_TEMPLATE.md` verbatim as the shape. Do not invent a different
body structure.

```markdown
## Summary

<a short paragraph - what changed and why>

## Changes
- <one bullet per logical change>

## Testing
- [ ] Built successfully
- [ ] Tests pass

## Related Issues
Closes #
```

Tick a checklist box only if it was actually verified this session. Leave the rest
unticked - do not tick optimistically. If you ran only the fast lane, say so under Testing
rather than ticking "Tests pass" unqualified.

## 4. Issue number

Look for it in the branch slug or the commit messages. If found, complete the `Closes #` line.
**If not found, do not guess a number** - leave `Closes #` bare as the template has it, and say
in your reply that no issue number was found.

## Output

A markdown title line, kept outside the fence since `gh pr create` takes `--title` and
`--body` as separate arguments, followed by the filled body inside its own single fenced
code block (` ```markdown `) so the body renders as literal, copy-pasteable text rather
than being rendered as chat formatting - ready to paste into `gh pr create --body`.
Nothing else - no push, no `gh pr create`.
