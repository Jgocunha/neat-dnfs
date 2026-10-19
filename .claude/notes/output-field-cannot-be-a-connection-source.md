# The output field cannot be a connection source

Evolution never creates a connection that leaves the output field:

- `Genome::isLegalConnectionSource` allows only input and hidden fields, so add-connection never picks the output as
  a source.
- Add-field only splits a connection that already exists, so it cannot create one from scratch either.

The template decoder (`Solution` building a genome from a `.dnf`) does not check this. A template with an
output-sourced connection loads, its offspring inherit the connection through crossover, and add-field can split it into
output → new field → target. It is never removed, but evolution could never have built it.

Since 19 October 2026:
- **Loading a template warns** (`27300248`): `makeFromTemplate`, which `--template` in `neat-dnfs-inc-evol` and
  `neat-dnfs-sol-eval` goes through, logs one WARNING per connection the rule forbids, naming both fields and their
  roles. `Genome::connectionTuplesEvolutionCannotMake()` lists them. The decoder does not warn, because it runs again
  for every copy a population makes of its starting solution.
- **The rule can be widened** (`7faae6f0`): with `AblationConstants.allowOutputAsConnectionSource` (preset
  `output-feedback`, data folder `<Task> Output Feedback`), mutation and seeding may also start a connection at an
  output field, to hidden fields and other outputs, never to itself. The key is optional and off when absent. In a
  genome with no hidden field and one output, a draw of the output as source finds no target and gives up, so
  1 in (inputs + 1) add-connection mutations of a minimal start does nothing.
  - `sol-eval` takes no `--ablation`, so it always warns under the default rule.

Hand-built designs that relied on one, found on 19 October 2026:

| Design | Connection | Role | Now |
|---|---|---|---|
| `tests/fixtures/serial-order-handbuilt.dnf` | `gk cg 2-7` | output → done field D | competition moved into a hidden selection field, which feeds D; the output copies it (`serial-order-handbuilt-design.md`). A test guards it |
| `serial-order-overlapping-peaks.dnf`, `serial-order-single-recall.dnf` | `gk cg 2-7` | as above | unchanged: failure-mode fixtures built from the earlier design |
| `tests/fixtures/sequence-timing-boost-handbuilt.dnf` | `gk cg 2-6` | output → past-events field E | E reads the integrator G instead (`gk cg 5-6`), below. A test guards it |
| bundle only: `sequence-timing/architectures/handbuilt-gaussian-x.dnf`, `handbuilt-boost-v1-fixture.dnf` | `gk cg 2-8`, `gk cg 2-6` | output → E | unchanged: superseded designs, not test fixtures. None of the seeds built from them has an E |

No genome evolved in the 56 saved runs (every sequence-timing and serial-order run in the experiment bundle) has a
connection out of field 2.

What this means:

- These hand-built designs show that DNFs can solve the tasks, not that NEAT can reach the solution from scratch.
- A seed that keeps such a connection starts from something evolution could not have built. A seed that removes the
  field it feeds asks evolution to rediscover that field by some other route.
- In sequence timing, a legal route exists and evolution found it. The boost b2 seed has no E. Run 18 solved it by
  adding `5-3`, from field 5, which drives the output, back to the memory: it erases each memory once it is recalled.
- In serial order the recalled item existed only in the output, where the competition happened. So the competition
  moved into a hidden selection field that the output copies, and the done field reads from that.
- In the boost design the output has no competition: each item's trace in G drives the output as it crosses threshold.
  So G already shows which item is being recalled, and E can read G directly. No new field is needed.

## Sequence timing (boost): the past-events field reads the integrator

19 October 2026. In `sequence-timing-boost-handbuilt.dnf`, `gk cg 2-6` (output → E, gk(2, 10)) became `gk cg 5-6`
(G → E, gk(2, 15)). Nothing else changed. Fifty evaluations each, against the design before:

| Task | Before | After |
|---|---|---|
| `sequence-timing-boost` | 0.988 | 0.988 |
| `sequence-timing-boost-order` | 0.991 | 0.991 |
| `sequence-timing-boost-order-only` | 0.993 | 0.993 |

In dnf-composer, at gaps 300, 700 and 1500, the output recalls D, C, A, B one at a time with the same onsets as
before (D 90-104, C 578-586, A 774-780, B 1750-1756 after X). Each peak lasts 6-10 longer, because E ignites a little
later from G's slower rise. E stays below threshold through the demonstration and the gap.

G->E works in a band about x2 wide at every width (`sequence-timing/architectures/reachable/sweep1-g-to-e.txt` in the
bundle, 10 evaluations each): 8-10 at width 1, 10-20 at width 2, 20-30 at width 4, 30-50 at width 6. Too weak, and
E never ignites, so the peaks pile up (0.878, f6 0.42). Too strong, and E ignites at the stored items before recall,
because G's traces sit just below threshold, and the output stays silent (0.328).

Moving the output's role into a hidden selection field, as in serial order, also works, but the extra copy delays every
peak: 0.952 on `sequence-timing-boost` (f5 0.86), untuned.
