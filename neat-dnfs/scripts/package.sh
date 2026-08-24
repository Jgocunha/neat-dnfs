#!/bin/bash
# Builds the release archive for this platform out of an existing Release build
# tree: the three executables plus the config/ and templates/ they resolve at
# startup (the "runtime" install component -- see CMakeLists.txt). Run
# ./scripts/build.sh (Linux) or ./scripts/build_macos.sh (macOS) first.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OS=$(uname -s)
if [ "$OS" = "Darwin" ]; then
    PLATFORM="macos"
    BUILD_DIR="$PROJECT_ROOT/build/macos-release"
else
    PLATFORM="linux"
    BUILD_DIR="$PROJECT_ROOT/build/linux-release"
fi

# uname spells the same architecture differently per platform; the archive name
# should not.
case "$(uname -m)" in
    x86_64|amd64) ARCH="x64" ;;
    arm64|aarch64) ARCH="arm64" ;;
    *) ARCH="$(uname -m)" ;;
esac

if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    echo "ERROR: no build found at $BUILD_DIR. Run ./scripts/build.sh first."
    exit 1
fi

VERSION=$(sed -n 's/^NEAT_DNFS_VERSION:STRING=//p' "$BUILD_DIR/CMakeCache.txt")
if [ -z "$VERSION" ]; then
    echo "ERROR: could not read NEAT_DNFS_VERSION from $BUILD_DIR/CMakeCache.txt."
    exit 1
fi

NAME="neat-dnfs-$VERSION-$PLATFORM-$ARCH"
PACKAGE_DIR="$PROJECT_ROOT/package"
STAGE="$PACKAGE_DIR/$NAME"

rm -rf "$STAGE" "$PACKAGE_DIR/$NAME.tar.gz"
mkdir -p "$PACKAGE_DIR"

cmake --install "$BUILD_DIR" --config Release --component runtime --prefix "$STAGE"

# -C so the archive holds a single top-level directory rather than spilling
# bin/ and share/ into whatever the user extracts it in.
tar -czf "$PACKAGE_DIR/$NAME.tar.gz" -C "$PACKAGE_DIR" "$NAME"

echo ""
echo "Package: $PACKAGE_DIR/$NAME.tar.gz"
