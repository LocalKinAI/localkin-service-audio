# OllamaAudio uv Setup Guide

## 🎉 OllamaAudio is now using uv!

Your project has been successfully converted to use `uv` for fast, reliable Python package management.

## 🚀 Quick Start

### Option 1: Use the Launcher Script

```bash
# Make the script executable (already done)
chmod +x ollamaaudio.sh

# Use ollamaaudio
./ollamaaudio.sh --help
./ollamaaudio.sh list
./ollamaaudio.sh stt audio.wav
```

### Option 2: Add to PATH (Recommended)

```bash
# Add pyenv bin to your PATH permanently
echo 'export PATH="$HOME/.pyenv/versions/3.10.0/bin:$PATH"' >> ~/.zshrc

# Or for bash
echo 'export PATH="$HOME/.pyenv/versions/3.10.0/bin:$PATH"' >> ~/.bashrc

# Reload your shell
source ~/.zshrc

# Now use ollamaaudio directly
ollamaaudio --version
ollamaaudio list
```

### Option 3: Create a symlink

```bash
# Create symlink to ollamaaudio in your local bin
ln -sf /Users/jackysun/.pyenv/versions/3.10.0/bin/ollamaaudio /usr/local/bin/ollamaaudio

# Or to your home bin
mkdir -p ~/bin
ln -sf /Users/jackysun/.pyenv/versions/3.10.0/bin/ollamaaudio ~/bin/ollamaaudio
export PATH="$HOME/bin:$PATH"
```

## 📦 What's New with uv

### ⚡ Performance Benefits
- **10-100x faster** than pip for dependency resolution
- **Parallel downloads** for faster installations
- **Smart caching** - dependencies only downloaded once

### 🛡️ Reliability
- **Reproducible builds** with lock files
- **Better dependency resolution** avoids conflicts
- **Atomic operations** - no partial installs

### 🛠️ Modern Python Packaging
- **pyproject.toml** as the single source of truth
- **Built-in tools** for linting, formatting, testing
- **Virtual environment management** built-in

## 🔧 Available Commands

```bash
# Package management
uv pip install -e .                    # Install in development mode
uv pip install -e ".[dev]"            # Install with dev dependencies
uv pip install -e ".[gpu]"            # Install with GPU support
uv venv                               # Create virtual environment
uv sync                               # Sync dependencies

# Development tools
uv run pytest                         # Run tests
uv run ruff check ollamaaudio/        # Lint code
uv run ruff format ollamaaudio/       # Format code
uv run mypy ollamaaudio/              # Type check
```

## 📁 Project Structure

```
ollamaaudio/
├── pyproject.toml           # ✨ New: Modern package configuration
├── setup.py                 # Updated: Minimal compatibility layer
├── ollamaaudio.sh           # ✨ New: Launcher script
├── ollamaaudio/
│   ├── cli.py              # Enhanced CLI
│   ├── config.py           # Configuration management
│   ├── models.py           # Ollama integration
│   ├── stt.py              # Speech-to-text
│   ├── tts.py              # Text-to-speech
│   └── models.json         # Model registry
└── README.md               # Updated with uv instructions
```

## 🎯 Why uv?

### Before (pip):
```bash
pip install -r requirements.txt  # Slow, potential conflicts
pip install -e .                # Manual dependency management
```

### After (uv):
```bash
uv pip install -e .             # Fast, reliable, conflict-free
uv sync                        # Perfect dependency resolution
```

### Performance Comparison:
- **Installation**: 10-100x faster
- **Dependency resolution**: Much more reliable
- **Caching**: Only download once, reuse everywhere

## 🔄 Migration Complete

✅ **pyproject.toml** - Modern configuration
✅ **uv installation** - Fast and reliable
✅ **Console scripts** - Direct command access
✅ **Updated docs** - uv instructions
✅ **Launcher script** - Easy access
✅ **All tests pass** - Functionality preserved

**Enjoy the speed and reliability of uv with OllamaAudio!** 🚀✨
