---
name: build-and-test
description: Configure, build and test neat-dnfs. Use whenever you need to compile the project, run the Catch2 suite, verify a change builds, or reproduce a test failure.
---

# Build and test

## Where things are

All commands run from the **nested project root**, `neat-dnfs/` inside the repository - not the
repository root. The repository root holds `README.md` and `.claude/`; `CMakeLists.txt` is one
level down.

**There is no `CMakePresets.json`.** The `scripts/` wrappers are the build path, and CI uses
them directly.

`VCPKG_ROOT` must be set in the environment - `CMakeLists.txt` fails outright without it, not
with a warning. If it is unset, run `scripts/setup.bat` (Windows) or `scripts/setup.sh`
(Linux/macOS) once. Beyond installing the vcpkg packages, setup clones and builds two sibling
dependencies into `deps/`:

| Dependency | Installs to |
|---|---|
| `imgui-platform-kit` | `deps/ipk-install` |
| `dynamic-neural-field-composer` | `deps/dnfc-install` |

Neither is found automatically - they reach CMake only through `-DCMAKE_PREFIX_PATH`, which the
build scripts pass for you. A hand-rolled `cmake -S . -B ...` that omits it fails at
`find_package`. This is the main reason to use the scripts rather than driving CMake yourself.

**Neither setup script leaves `VCPKG_ROOT` set in the shell you ran it from** - `setup.sh` only
`export`s inside its own process, and `setup.bat` writes it via `setx`, which only reaches
shells opened afterward. Open a new shell before building, or set the variable directly for the
current session.

## Build

The scripts configure and build in one step:

```bat
scripts\build.bat
```
```bash
./scripts/build.sh          # linux
./scripts/build_macos.sh    # macos
```

They write to **different trees**, which is worth knowing before you go looking for a binary:

| Script | Tree |
|---|---|
| `scripts/build.bat` | `build/x64-release` **and** `build/x64-debug` |
| `scripts/build.sh` | `build/linux-release` |
| `scripts/build_macos.sh` | `build/macos-release` |

`build.bat` builds Release *and* Debug by default. Set `NEAT_DNFS_RELEASE_ONLY=1` to skip the
Debug leg - CI does, and it roughly halves the time.

Once a tree exists, build a single target directly rather than re-running the script:

```bash
cmake --build build/x64-release --config Release --target neat-dnfs-test --parallel 4
```

**Cap parallelism at 4.** Several agents may be building on the same machine at once;
unbounded `--parallel` oversubscribes every core and slows all of them down.

Targets: `neat-dnfs-test`, `neat-dnfs` (the library), `neat-dnfs-evol`, `neat-dnfs-inc-evol`,
`neat-dnfs-sol-eval`.

A cold build takes several minutes - run it in the background rather than blocking on a
foreground timeout.

Sanitizer builds: configure with `-DNEAT_DNFS_SANITIZER=address` or `=thread`.

## Test

Run `ctest` from inside the build tree. **There are two lanes**, and picking the wrong one is
the most common mistake here:

```bash
ctest -C Release -LE slow --output-on-failure --parallel   # fast lane - 196 tests, <10s
ctest -C Release -L  slow --output-on-failure --parallel   # [Evolution] tier - tens of seconds
ctest -C Release        --output-on-failure --parallel     # everything
```

`-C Release` is required on Windows (multi-config); harmless elsewhere.

The `slow` lane runs real `Population::evolve()` at populationSize=50 / numGenerations=10 /
numRuns=5 per solution. It is **stochastic** - re-run, and reproduce on unmodified `main`,
before reporting a failure there as a regression.
PR runs on Windows and macOS use `-LE slow`, so only Linux exercises that tier in CI.

**On Windows, put the vcpkg DLLs on `PATH` first** or the binary fails to start rather than
failing a test:

```bat
set "PATH=%VCPKG_ROOT%/installed/x64-windows/bin;%PATH%"
```

For the TDD inner loop, run the Catch2 binary directly with a tag or name filter:

```bash
./build/x64-release/Release/neat-dnfs-test.exe "[Population]"
./build/x64-release/Release/neat-dnfs-test.exe --list-tests
./build/x64-release/Release/neat-dnfs-test.exe "[Evolution]" --order rand --rng-seed 12345
```

Record the seed whenever you use `--order rand` - without it the run is not reproducible.

## Python analysis suite

The Streamlit visualizer has its own tests, run separately from CTest and covered by the
`analysis-tests` CI job:

```bash
cd analysis && python -m pytest
```

## Adding a test file

`CMakeLists.txt` lists every test source **explicitly**. There is no `GLOB` anywhere in this
project, so a new file under `tests/` that isn't added to that list is silently never compiled,
and the suite goes green having never run it. Add it, then reconfigure before building.

Check this whenever a new test "passes" on its first run - that is the symptom.

## Reporting results

Report what actually happened, with real output:

- Green: say so plainly, with the test count, **and which lane you ran**. "Tests pass" after
  `-LE slow` is a different claim from "tests pass", and the difference matters.
- Red: paste the failing assertion and the test name. Do not summarise a failure as "some
  tests failed" - the actual message is the useful part.
- **Check the build's own exit code, not a pipeline's.** Piping cmake into `tail` or
  `Select-Object` returns the *pipe's* status, so a failed build reports success. Capture
  the exit code directly, or scan the output for `FAILED:`.
- Do not claim verification you did not perform. If you only built and did not run the
  suite, say that.
