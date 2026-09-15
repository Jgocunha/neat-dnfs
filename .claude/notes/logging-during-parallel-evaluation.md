# Logging during parallel solution evaluation crashes

## Symptom

Intermittent `SegFault`s in `Population` tests (`ctest -LE slow`), most reliably in the
largest runs - e.g. *"the recorded best never falls below the established high-water mark"*
(population 50 x 15 generations). Re-running the failed tests serially passes most of them,
so it reads like flakiness rather than a bug.

## Cause

`PopulationParameters::parallelEvolution` evaluates solutions on `std::async` threads, so any
log call inside fitness code (`testPhenotype()` or a `Solution` fitness helper) runs concurrently.

`neat_dnfs::tools::logger::log` with the default `LogOutputMode::ALL` writes to two sinks:

| Sink | Thread-safe? |
|---|---|
| `Logger::log_cmd` - stdout | yes, `static std::mutex` around `std::cout` |
| `Logger::log_ui` -> `imgui_kit::LogWindow::addLog` | **no** - appends to an unguarded `inline static std::vector<LogEntry> logs` |

Concurrent `addLog` calls race on that vector, corrupt the heap, and crash.

`tests/entry.cpp` already documents the same class of race for *dnf_composer*'s logger and
raises it to `ERROR`. neat-dnfs's own logger is not raised, and its minimum level defaults to
`DEBUG`.

## What to do

- Don't log from fitness code on the evaluation hot path. For temporary instrumentation, prefer
  running `neat-dnfs-sol-eval` (single-threaded) over the test suite or `neat-dnfs-evol`.
- `DEBUG` is **compiled out of Release** builds (`#ifndef _DEBUG` in `logger.cpp`), so the
  CLAUDE.md "log at DEBUG" advice is invisible in `build/x64-release`. Temporary INFO logging
  works there, but is exactly what triggers this crash under parallel evaluation.
- If logging from fitness code is ever needed permanently, pass `LogOutputMode::CONSOLE` (stdout
  only, mutex-guarded), or add a lock around `LogWindow::addLog`.
