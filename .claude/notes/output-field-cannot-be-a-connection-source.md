# The output field cannot be a connection source

Evolution never creates a connection that leaves the output field:

- `Genome::isLegalConnectionSource` allows only input and hidden fields, so add-connection never picks the output as
  a source.
- Add-field only splits a connection that already exists, so it cannot create one from scratch either.

The template decoder (`Solution` building a genome from a `.dnf`) does not check this. A template with an
output-sourced connection loads, its offspring inherit the connection through crossover, and add-field can split it into
output → new field → target. It is never removed, but evolution could never have built it.

Hand-built designs that relied on one, found on 19 October 2026:

| Design | Connection | Role | Now |
|---|---|---|---|
| `tests/fixtures/serial-order-handbuilt.dnf` | `gk cg 2-7` | output → done field D | competition moved into a hidden selection field, which feeds D; the output copies it (`serial-order-handbuilt-design.md`). A test guards it |
| `serial-order-overlapping-peaks.dnf`, `serial-order-single-recall.dnf` | `gk cg 2-7` | as above | unchanged: failure-mode fixtures built from the earlier design |
| `tests/fixtures/sequence-timing-boost-handbuilt.dnf` | `gk cg 2-6` | output → past-events field E | unchanged |

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
