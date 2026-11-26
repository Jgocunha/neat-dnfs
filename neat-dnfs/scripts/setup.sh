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

# ── vcpkg baseline ────────────────────────────────────────────────────────────
# Every vcpkg dependency, imgui-platform-kit included, is declared in vcpkg.json and
# installed by CMake on the first configure, at the versions of the vcpkg commit pinned
# there as builtin-baseline. vcpkg reads those versions from the files it has checked
# out, so VCPKG_ROOT has to be a full clone at that commit or a newer one.
BASELINE=$(sed -n 's/.*"builtin-baseline": *"\([0-9a-f]*\)".*/\1/p' "$PROJECT_ROOT/vcpkg.json")
if [ -z "$BASELINE" ]; then
    echo "ERROR: no builtin-baseline found in $PROJECT_ROOT/vcpkg.json."
    exit 1
fi
if [ "$(git -C "$VCPKG_ROOT" rev-parse --is-shallow-repository 2>/dev/null)" = "true" ]; then
    echo "vcpkg at $VCPKG_ROOT is a shallow clone, fetching its full history..."
    git -C "$VCPKG_ROOT" fetch --unshallow origin || {
        echo "ERROR: could not fetch the history of $VCPKG_ROOT. VCPKG_ROOT must be a clone you can write to."
        exit 1
    }
fi
if ! git -C "$VCPKG_ROOT" merge-base --is-ancestor "$BASELINE" HEAD 2>/dev/null; then
    echo "ERROR: vcpkg at $VCPKG_ROOT is older than the pinned baseline $BASELINE, or is not a git clone."
    echo "VCPKG_ROOT must be a git clone of https://github.com/microsoft/vcpkg at that commit or newer."
    echo "To move it to the pinned baseline:"
    echo "  git -C \"$VCPKG_ROOT\" fetch origin && git -C \"$VCPKG_ROOT\" checkout $BASELINE && \"$VCPKG_ROOT/bootstrap-vcpkg.sh\" -disableMetrics"
    exit 1
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

# ── helpers ────────────────────────────────────────────────────────────────────
# Clones SRC from URL if missing, then checks out REF -- independent of whether the
# clone just happened, so a source directory left over from a previous run still ends
# up on the right revision. Fetches first when an older clone lacks REF.
clone_at_ref() {
    local src="$1" url="$2" ref="$3"
    if [ ! -d "$src" ]; then
        echo "Cloning $(basename "$src")..."
        git clone "$url" "$src"
    fi
    if ! git -C "$src" cat-file -e "$ref^{commit}" 2>/dev/null; then
        git -C "$src" fetch origin
    fi
    git -C "$src" checkout "$ref"
}

# Removes INSTALL_DIR if it was built from anything other than STAMP, so the caller's
# own "already installed" check rebuilds instead of silently reusing a stale build.
invalidate_if_stale() {
    local install_dir="$1" stamp="$2" revision_file="$1/.revision"
    if [ -d "$install_dir" ] && [ "$(cat "$revision_file" 2>/dev/null)" != "$stamp" ]; then
        echo "$(basename "$install_dir") was built from a different revision or dependency set, rebuilding."
        rm -rf "$install_dir"
    fi
}

# ── dynamic-neural-field-composer ─────────────────────────────────────────────
# Pinned by scripts/dnfc-ref.txt; DNFC_REF overrides it. It is built against this
# project's vcpkg.json (VCPKG_MANIFEST_DIR), so the library and this project link the
# same dependency versions. The install is stamped with the revision and a hash of the
# manifest, so changing either one rebuilds it.
DNFC_SRC="$DEPS_DIR/dynamic-neural-field-composer"
DNFC_INSTALL="$DEPS_DIR/dnfc-install"
DNFC_REF="${DNFC_REF:-$(cat "$SCRIPT_DIR/dnfc-ref.txt")}"
MANIFEST_HASH=$(cat "$PROJECT_ROOT/vcpkg.json" "$PROJECT_ROOT/vcpkg-configuration.json" "$PROJECT_ROOT"/ports/*/* | git hash-object --stdin)
DNFC_STAMP="$DNFC_REF $MANIFEST_HASH"
PARALLEL=$( [ "$OS" = "Darwin" ] && sysctl -n hw.logicalcpu || nproc )

invalidate_if_stale "$DNFC_INSTALL" "$DNFC_STAMP"

if [ ! -d "$DNFC_INSTALL" ]; then
    clone_at_ref "$DNFC_SRC" https://github.com/Jgocunha/dynamic-neural-field-composer.git "$DNFC_REF"
    echo "Building dynamic-neural-field-composer..."
    rm -rf "$DNFC_SRC/build"
    cmake -S "$DNFC_SRC/dynamic-neural-field-composer" -B "$DNFC_SRC/build" \
        -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
        -DVCPKG_TARGET_TRIPLET="$TRIPLET" \
        -DVCPKG_MANIFEST_DIR="$PROJECT_ROOT" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$DNFC_INSTALL" \
        -DDNF_COMPOSER_BUILD_TESTS=OFF
    cmake --build "$DNFC_SRC/build" --parallel "$PARALLEL"
    cmake --install "$DNFC_SRC/build"
    echo "$DNFC_STAMP" > "$DNFC_INSTALL/.revision"
else
    echo "dynamic-neural-field-composer already installed, skipping."
fi

echo ""
echo "Setup complete. Run ./scripts/build.sh (Linux) or ./scripts/build_macos.sh (macOS) to build the project."
