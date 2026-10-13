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
PARALLEL=$( [ "$OS" = "Darwin" ] && sysctl -n hw.logicalcpu || nproc )

# ── vcpkg packages ────────────────────────────────────────────────────────────
echo "Installing vcpkg packages..."
"$VCPKG_ROOT/vcpkg" install \
    "imgui[docking-experimental,core,opengl3-binding,glfw-binding]:$TRIPLET" \
    "implot:$TRIPLET" \
    "imgui-node-editor:$TRIPLET" \
    "nlohmann-json:$TRIPLET" \
    "catch2:$TRIPLET" \
    "fftw3:$TRIPLET"

# ── helpers ────────────────────────────────────────────────────────────────────
# Clones SRC from URL if missing, then checks out REF if one was requested --
# independent of whether the clone just happened, so a source directory left
# over from a previous run still ends up on the right revision.
clone_at_ref() {
    local src="$1" url="$2" ref="$3"
    if [ ! -d "$src" ]; then
        echo "Cloning $(basename "$src")..."
        git clone "$url" "$src"
    fi
    if [ -n "$ref" ]; then
        git -C "$src" checkout "$ref"
    fi
}

# Removes INSTALL_DIR if it was built from a different revision than REF, so
# the caller's own "already installed" check rebuilds instead of silently
# reusing a stale revision.
invalidate_if_stale() {
    local install_dir="$1" ref="$2" revision_file="$1/.revision"
    if [ -n "$ref" ] && [ -d "$install_dir" ] && [ "$(cat "$revision_file" 2>/dev/null)" != "$ref" ]; then
        echo "$(basename "$install_dir") was built from a different revision, rebuilding."
        rm -rf "$install_dir"
    fi
}

# Which configuration(s) to build. Release only by default, matching
# scripts/build.sh; NEAT_DNFS_RELEASE_ONLY=1 forces Release only regardless of
# the argument, matching the env var's meaning in scripts/setup.bat and
# scripts/build.bat.
CONFIG="${1:-release}"
case "$CONFIG" in
    release|debug|all) ;;
    *)
        echo "ERROR: unknown setup configuration \"$CONFIG\" (expected \"release\", \"debug\", \"all\", or no argument for release)."
        exit 1
        ;;
esac
if [ "$NEAT_DNFS_RELEASE_ONLY" = "1" ]; then
    CONFIG=release
fi
[ "$CONFIG" = all ] && CONFIGS="release debug" || CONFIGS="$CONFIG"

# Migration: before per-configuration installs, deps/ipk-install/ and
# deps/dnfc-install/ each held a single flat Release tree. deps/ is
# gitignored, so a developer pulling this change keeps that flat tree; the
# per-configuration guards below would see the directory exist, skip the
# build entirely, and the project build would then fail at find_package with
# nothing pointing at why. .revision cannot catch this -- it only fires when
# IPK_REF/DNFC_REF are set, i.e. in CI. A flat tree is recognised by share/
# sitting directly under the install root; in the new layout share/ only ever
# exists one level down, under release/ or debug/.
migrate_flat_layout() {
    local install_dir="$1"
    if [ -d "$install_dir/share" ]; then
        echo "$(basename "$install_dir")/ uses the pre-per-configuration flat layout; removing it so it can be rebuilt per configuration."
        rm -rf "$install_dir"
    fi
}

# ── imgui-platform-kit ────────────────────────────────────────────────────────
IPK_SRC="$DEPS_DIR/imgui-platform-kit"
IPK_INSTALL="$DEPS_DIR/ipk-install"
IPK_URL=https://github.com/Jgocunha/imgui-platform-kit.git

migrate_flat_layout "$IPK_INSTALL"

for cfg in $CONFIGS; do
    [ "$cfg" = debug ] && cmake_cfg=Debug || cmake_cfg=Release
    invalidate_if_stale "$IPK_INSTALL/$cfg" "$IPK_REF"
    if [ -d "$IPK_INSTALL/$cfg" ]; then
        echo "imgui-platform-kit $cfg already installed, skipping."
        continue
    fi
    clone_at_ref "$IPK_SRC" "$IPK_URL" "$IPK_REF"
    echo "Building imgui-platform-kit $cfg..."
    cmake -S "$IPK_SRC/imgui-platform-kit" -B "$IPK_SRC/build-$cfg" \
        -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
        -DVCPKG_TARGET_TRIPLET="$TRIPLET" \
        -DCMAKE_BUILD_TYPE="$cmake_cfg" \
        -DCMAKE_INSTALL_PREFIX="$IPK_INSTALL/$cfg"
    cmake --build "$IPK_SRC/build-$cfg" --parallel "$PARALLEL"
    cmake --install "$IPK_SRC/build-$cfg"
    [ -n "$IPK_REF" ] && echo "$IPK_REF" > "$IPK_INSTALL/$cfg/.revision"
done

# ── dynamic-neural-field-composer ─────────────────────────────────────────────
DNFC_SRC="$DEPS_DIR/dynamic-neural-field-composer"
DNFC_INSTALL="$DEPS_DIR/dnfc-install"
DNFC_URL=https://github.com/Jgocunha/dynamic-neural-field-composer.git

migrate_flat_layout "$DNFC_INSTALL"

for cfg in $CONFIGS; do
    [ "$cfg" = debug ] && cmake_cfg=Debug || cmake_cfg=Release
    invalidate_if_stale "$DNFC_INSTALL/$cfg" "$DNFC_REF"
    if [ -d "$DNFC_INSTALL/$cfg" ]; then
        echo "dynamic-neural-field-composer $cfg already installed, skipping."
        continue
    fi
    clone_at_ref "$DNFC_SRC" "$DNFC_URL" "$DNFC_REF"
    echo "Building dynamic-neural-field-composer $cfg..."
    cmake -S "$DNFC_SRC/dynamic-neural-field-composer" -B "$DNFC_SRC/build-$cfg" \
        -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
        -DVCPKG_TARGET_TRIPLET="$TRIPLET" \
        -DCMAKE_BUILD_TYPE="$cmake_cfg" \
        -DCMAKE_PREFIX_PATH="$IPK_INSTALL/$cfg" \
        -DCMAKE_INSTALL_PREFIX="$DNFC_INSTALL/$cfg" \
        -DDNF_COMPOSER_BUILD_TESTS=OFF
    cmake --build "$DNFC_SRC/build-$cfg" --parallel "$PARALLEL"
    cmake --install "$DNFC_SRC/build-$cfg"
    [ -n "$DNFC_REF" ] && echo "$DNFC_REF" > "$DNFC_INSTALL/$cfg/.revision"
done

echo ""
echo "Setup complete. Run ./scripts/build.sh (Linux) or ./scripts/build_macos.sh (macOS) to build the project."
