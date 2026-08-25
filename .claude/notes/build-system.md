# Build system

## There is no `CMakePresets.json`

The `scripts/` wrappers are the build path, and CI uses them directly. There is exactly one
`CMakeLists.txt`, at `neat-dnfs/CMakeLists.txt` - no subdirectory CMake files.

## `VCPKG_ROOT` is a hard failure, not a warning

```cmake
if(NOT DEFINED ENV{VCPKG_ROOT})
    message(FATAL_ERROR "ERROR: This project requires VCPKG.\n")
endif()
```

vcpkg is used in *classic* mode - there is no `vcpkg.json` manifest. Packages are installed
imperatively by `scripts/setup.{bat,sh}`: `imgui[docking-experimental,core,opengl3-binding,glfw-binding]`,
`implot`, `imgui-node-editor`, `nlohmann-json`, `catch2`, `fftw3`.

## Two sibling dependencies are built from source into `deps/`

`scripts/setup.*` also clones and builds two of the author's own repos:

| Dependency | Installs to |
|---|---|
| `imgui-platform-kit` | `neat-dnfs/deps/ipk-install` |
| `dynamic-neural-field-composer` | `neat-dnfs/deps/dnfc-install` |

These are not found automatically. They reach CMake only via `-DCMAKE_PREFIX_PATH`, which the
build scripts pass. A hand-rolled `cmake -S . -B ...` that omits it fails at `find_package`.

## The three scripts build three differently-named trees

| Script | Tree |
|---|---|
| `scripts/build.bat` | `build/x64-release` and `build/x64-debug` |
| `scripts/build.sh` | `build/linux-release` |
| `scripts/build_macos.sh` | `build/macos-release` |

`build.bat` builds Release *and* Debug by default. `NEAT_DNFS_RELEASE_ONLY=1` skips the Debug
leg - CI sets this, and it roughly halves Windows build time.

The Windows configure also passes `-DCMAKE_MAP_IMPORTED_CONFIG_MINSIZEREL=Release` and
`-DCMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO=Release`, so imported dependency targets resolve in
those configs.

## Sanitizers

`-DNEAT_DNFS_SANITIZER=address` or `=thread`. The `sanitize` CI job runs both.

Note that `Population` evolves in parallel via `std::async` and
`PopulationParameters::parallelEvolution` defaults to `true`, so the TSan job exercises a
genuinely concurrent path - including code inside the upstream `dnf_composer` dependency.

## Running the binary needs the vcpkg DLLs on Windows

```bat
set "PATH=%VCPKG_ROOT%/installed/x64-windows/bin;%PATH%"
```

Without it the test executable fails to start rather than failing a test.
