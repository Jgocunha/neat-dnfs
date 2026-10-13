#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPS_DIR="$PROJECT_ROOT/deps"
IPK_INSTALL="$DEPS_DIR/ipk-install"
DNFC_INSTALL="$DEPS_DIR/dnfc-install"

if [ -z "$VCPKG_ROOT" ]; then
    echo "ERROR: The environment variable VCPKG_ROOT is not set."
    echo "Run ./scripts/setup.sh first to install all dependencies automatically."
    exit 1
fi

ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    VCPKG_TRIPLET="arm64-osx"
else
    VCPKG_TRIPLET="x64-osx"
fi

echo "Detected architecture: $ARCH, using vcpkg triplet: $VCPKG_TRIPLET"

# Which configuration(s) to build. Release only by default, matching
# scripts/setup.sh's own default.
CONFIG="${1:-release}"
case "$CONFIG" in
    release|debug|all) ;;
    *)
        echo "ERROR: unknown build configuration \"$CONFIG\" (expected \"release\", \"debug\", \"all\", or no argument for release)."
        exit 1
        ;;
esac
[ "$CONFIG" = all ] && CONFIGS="release debug" || CONFIGS="$CONFIG"

PARALLEL="$(sysctl -n hw.logicalcpu)"

for cfg in $CONFIGS; do
    [ "$cfg" = debug ] && cmake_cfg=Debug || cmake_cfg=Release
    ipk="$IPK_INSTALL/$cfg"
    dnfc="$DNFC_INSTALL/$cfg"
    # deps/ipk-install and deps/dnfc-install used to be single flat Release
    # trees. A developer who pulled that change but has not re-run setup.sh
    # would otherwise fail deep inside find_package with no hint about what
    # to do.
    if [ ! -d "$ipk" ] || [ ! -d "$dnfc" ]; then
        echo "ERROR: no $cfg install at $ipk and/or $dnfc."
        echo "Run ./scripts/setup.sh $cfg (or ./scripts/setup.sh all) first."
        exit 1
    fi
    build_dir="$PROJECT_ROOT/build/macos-$cfg"
    mkdir -p "$build_dir"
    cmake -S "$PROJECT_ROOT" -B "$build_dir" \
        -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
        -DVCPKG_TARGET_TRIPLET="$VCPKG_TRIPLET" \
        -DCMAKE_BUILD_TYPE="$cmake_cfg" \
        -DCMAKE_PREFIX_PATH="$ipk;$dnfc"
    cmake --build "$build_dir" --parallel "$PARALLEL"
done
