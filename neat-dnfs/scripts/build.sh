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

BUILD_DIR="$PROJECT_ROOT/build/linux-release"
mkdir -p "$BUILD_DIR"

cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" \
    -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$IPK_INSTALL;$DNFC_INSTALL"

cmake --build "$BUILD_DIR" --parallel "$(nproc)"
