#!/bin/bash
# =============================================================================
# CoCoSpec Project - Linux Main Script with Auto-Troubleshooting
# =============================================================================

set -e

# Project configuration
PROJECT_NAME="GBM_Detection"
ENV_NAME="cocospec_env"

# === Python version auto-detection (3.8 to 3.12, lowest available) ===
PYTHON_EXE=""
for V in 3.8 3.9 3.10 3.11 3.12; do
    if command -v python$V >/dev/null 2>&1; then
        PYTHON_EXE="python$V"
        break
    fi
done
if [ -z "$PYTHON_EXE" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_EXE="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_EXE="python"
    else
        echo "Error: No suitable Python 3.8-3.12 found in PATH."
        echo "Please install Python 3.8, 3.9, 3.10, 3.11 or 3.12 and add it to your PATH."
        exit 1
    fi
fi
PYTHON_PATH=$($PYTHON_EXE -c "import sys; print(sys.executable)")
echo "Selected Python: $PYTHON_PATH"

echo "============================================"
echo "  $PROJECT_NAME Project Setup & Run"
echo "============================================"

# Main execution starts here
# ...existing code for environment setup, extraction, etc...
