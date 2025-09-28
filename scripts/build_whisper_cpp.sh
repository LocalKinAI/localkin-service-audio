#!/bin/bash
# Build script for whisper.cpp integration with LocalKin Service Audio

set -e

echo "🎵 LocalKin Service Audio - whisper.cpp Build Script"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Set default values
WHISPER_CPP_VERSION=${WHISPER_CPP_VERSION:-"master"}
BUILD_TYPE=${BUILD_TYPE:-"Release"}
INSTALL_PREFIX=${INSTALL_PREFIX:-"$HOME/.local"}

echo "🔧 Build configuration:"
echo "  Version: $WHISPER_CPP_VERSION"
echo "  Build type: $BUILD_TYPE"
echo "  Install prefix: $INSTALL_PREFIX"
echo ""

# Check for required tools
echo "🔍 Checking for required tools..."

if ! command -v git &> /dev/null; then
    echo "❌ git is required but not installed"
    exit 1
fi

if ! command -v make &> /dev/null; then
    echo "❌ make is required but not installed"
    exit 1
fi

if ! command -v cmake &> /dev/null; then
    echo "❌ cmake is required but not installed"
    exit 1
fi

echo "✅ All required tools found"
echo ""

# Clone or update whisper.cpp
WHISPER_CPP_DIR="external/whisper.cpp"

if [ -d "$WHISPER_CPP_DIR" ]; then
    echo "🔄 Updating existing whisper.cpp repository..."
    cd "$WHISPER_CPP_DIR"
    git fetch origin
    git checkout "$WHISPER_CPP_VERSION"
    cd ../..
else
    echo "📥 Cloning whisper.cpp repository..."
    mkdir -p external
    git clone https://github.com/ggerganov/whisper.cpp.git "$WHISPER_CPP_DIR"
    cd "$WHISPER_CPP_DIR"
    git checkout "$WHISPER_CPP_VERSION"
    cd ../..
fi

echo "✅ whisper.cpp repository ready"
echo ""

# Build whisper.cpp
echo "🔨 Building whisper.cpp..."

cd "$WHISPER_CPP_DIR"

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "⚙️  Configuring build..."
cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_INSTALL_RPATH="$INSTALL_PREFIX/lib" \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE \
    -DWHISPER_BUILD_TESTS=OFF \
    -DWHISPER_BUILD_EXAMPLES=ON \
    -DWHISPER_SUPPORT_OPENBLAS=ON \
    -DWHISPER_SUPPORT_SDL2=OFF

# Build
echo "🔨 Building..."
make -j$(nproc)

# Install
echo "📦 Installing..."
make install

cd ../../..

# Check if installation was successful
if [ -f "$INSTALL_PREFIX/bin/whisper-cli" ]; then
    echo "✅ whisper.cpp installed successfully!"
    echo "📍 Executable: $INSTALL_PREFIX/bin/whisper-cli"

    # Add to PATH if not already there
    if [[ ":$PATH:" != *":$INSTALL_PREFIX/bin:"* ]]; then
        echo "💡 Add to PATH: export PATH=\"$INSTALL_PREFIX/bin:\$PATH\""
        echo "   Or create symlink: sudo ln -sf $INSTALL_PREFIX/bin/whisper-cli /usr/local/bin/"
    fi

    echo ""
    echo "🎯 Next steps:"
    echo "1. Download models: python scripts/download_whisper_cpp_models.py tiny base"
    echo "2. Test integration: kin audio transcribe audio.wav --engine whisper-cpp --model_size tiny"
    echo "3. Run server: kin audio run whisper-cpp-tiny --port 8000"
else
    echo "❌ Installation failed"
    exit 1
fi

echo ""
echo "🎉 whisper.cpp integration complete!"
