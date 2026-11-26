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

vcpkg runs in *manifest* mode. `neat-dnfs/vcpkg.json` declares every vcpkg dependency and pins
its version with `builtin-baseline` (one vcpkg commit); the first configure of a build tree
installs them into `<tree>/vcpkg_installed/`. `imgui-platform-kit` is an overlay port in
`ports/`, pinned by `REF`/`SHA512` in its `portfile.cmake`. `VCPKG_ROOT` must be a git clone
containing the baseline commit; `scripts/setup.*` fetch it if missing.

## dynamic-neural-field-composer is built from source into `deps/`

`scripts/setup.*` clone it at the commit in `scripts/dnfc-ref.txt` (`DNFC_REF` overrides it)
and install it into `neat-dnfs/deps/dnfc-install`. It is configured with
`-DVCPKG_MANIFEST_DIR=<neat-dnfs>`, so it compiles against **this** project's manifest - one
dependency set for both, no second imgui. Its install is stamped (`.revision`) with the ref
and a hash of `vcpkg.json`/`vcpkg-configuration.json`/`ports/`; changing any of them makes the
next setup rebuild it. It stays source-built rather than a vcpkg port because it reads its
fonts and `data/` from its own source tree at runtime (compiled-in `PROJECT_DIR` fallback).

It is not found automatically. It reaches CMake only via `-DCMAKE_PREFIX_PATH`, which the
build scripts pass. A hand-rolled `cmake -S . -B ...` that omits it fails at `find_package`.

Build provenance (`build_info.h`) reads the kit's revision from the port's `REF`, and the
package list from `vcpkg list --x-install-root=${VCPKG_INSTALLED_DIR}` - the build tree's
manifest install, not vcpkg's classic `installed/`.

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
