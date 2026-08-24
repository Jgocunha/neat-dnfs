#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPS_DIR="$PROJECT_ROOT/deps"

# ── vcpkg ─────────────────────────────────────────────────────────────────────
if [ -z "$VCPKG_ROOT" ]; then
    export VCPKG_ROOT="$HOME/vcpkg"
    echo "VCPKG_ROOT not set. Installing vcpkg to $VCPKG_ROOT..."
    if [ ! -d "$VCPKG_ROOT" ]; then
        git clone https://github.com/microsoft/vcpkg.git "$VCPKG_ROOT"
        "$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics
    fi
    echo ""
    echo "Add the following line to your shell profile (~/.bashrc or ~/.zshrc) to persist VCPKG_ROOT:"
    echo "  export VCPKG_ROOT=$VCPKG_ROOT"
    echo ""
fi

# ── triplet detection ─────────────────────────────────────────────────────────
OS=$(uname -s)
ARCH=$(uname -m)
if [ "$OS" = "Darwin" ]; then
    TRIPLET=$( [ "$ARCH" = "arm64" ] && echo "arm64-osx" || echo "x64-osx" )
else
    TRIPLET="x64-linux"
fi
echo "Using vcpkg triplet: $TRIPLET"

# ── vcpkg packages ────────────────────────────────────────────────────────────
echo "Installing vcpkg packages..."
"$VCPKG_ROOT/vcpkg" install \
    "imgui[docking-experimental,core,opengl3-binding,glfw-binding]:$TRIPLET" \
    "implot:$TRIPLET" \
    "imgui-node-editor:$TRIPLET" \
    "nlohmann-json:$TRIPLET" \
    "catch2:$TRIPLET" \
    "gtest:$TRIPLET" \
    "fftw3:$TRIPLET"

# ── imgui-platform-kit ────────────────────────────────────────────────────────
IPK_SRC="$DEPS_DIR/imgui-platform-kit"
IPK_INSTALL="$DEPS_DIR/ipk-install"
PARALLEL=$( [ "$OS" = "Darwin" ] && sysctl -n hw.logicalcpu || nproc )

if [ ! -d "$IPK_SRC" ]; then
    echo "Cloning imgui-platform-kit..."
    git clone https://github.com/Jgocunha/imgui-platform-kit.git "$IPK_SRC"
fi

if [ ! -d "$IPK_INSTALL" ]; then
    echo "Building imgui-platform-kit..."
    cmake -S "$IPK_SRC/imgui-platform-kit" -B "$IPK_SRC/build" \
        -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
        -DVCPKG_TARGET_TRIPLET="$TRIPLET" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$IPK_INSTALL"
    cmake --build "$IPK_SRC/build" --parallel "$PARALLEL"
    cmake --install "$IPK_SRC/build"
else
    echo "imgui-platform-kit already installed, skipping."
fi

# ── dynamic-neural-field-composer ─────────────────────────────────────────────
DNFC_SRC="$DEPS_DIR/dynamic-neural-field-composer"
DNFC_INSTALL="$DEPS_DIR/dnfc-install"

if [ ! -d "$DNFC_SRC" ]; then
    echo "Cloning dynamic-neural-field-composer..."
    git clone https://github.com/Jgocunha/dynamic-neural-field-composer.git "$DNFC_SRC"
fi

if [ ! -d "$DNFC_INSTALL" ]; then
    echo "Building dynamic-neural-field-composer..."
    cmake -S "$DNFC_SRC/dynamic-neural-field-composer" -B "$DNFC_SRC/build" \
        -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
        -DVCPKG_TARGET_TRIPLET="$TRIPLET" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$IPK_INSTALL" \
        -DCMAKE_INSTALL_PREFIX="$DNFC_INSTALL" \
        -DDNF_COMPOSER_BUILD_TESTS=OFF
    cmake --build "$DNFC_SRC/build" --parallel "$PARALLEL"
    cmake --install "$DNFC_SRC/build"
else
    echo "dynamic-neural-field-composer already installed, skipping."
fi

echo ""
echo "Setup complete. Run ./scripts/build.sh (Linux) or ./scripts/build_macos.sh (macOS) to build the project."
