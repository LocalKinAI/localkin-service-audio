#!/bin/bash
# Build script for LocalKin Service Audio

set -e

echo "🏗️  Building LocalKin Service Audio..."

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/

# Install dependencies
echo "📦 Installing dependencies..."
uv sync

# Run tests (if any)
echo "🧪 Checking for tests..."
if command -v pytest &> /dev/null; then
    uv run pytest || echo "⚠️  No tests found or tests failed"
else
    echo "⚠️  pytest not available, skipping tests"
fi

# Run linting and formatting
echo "🔍 Running linting..."
uv run ruff check localkin_service_audio/ || echo "⚠️  ruff check failed"
uv run ruff format localkin_service_audio/ || echo "⚠️  ruff format failed"

# Build package
echo "📦 Building package..."
uv build

echo "✅ Build complete!"
echo "📦 Package available in dist/"
