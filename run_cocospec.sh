#!/bin/bash
# =============================================================================
# CoCoSpec Project - Linux Main Script with Auto-Troubleshooting
# =============================================================================

set -e

# Project configuration
PROJECT_NAME="GBM_Detection"
ENV_NAME="cocospec_env"

done

# === Python version check: ONLY Python 3.12.3 is supported ===
PYTHON_EXE=""
if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_EXE="python3.12"
elif command -v python3 >/dev/null 2>&1; then
    VER=$(python3 --version 2>&1)
    if [[ "$VER" == *"3.12.3"* ]]; then
        PYTHON_EXE="python3"
    fi
elif command -v python >/dev/null 2>&1; then
    VER=$(python --version 2>&1)
    if [[ "$VER" == *"3.12.3"* ]]; then
        PYTHON_EXE="python"
    fi
fi
if [ -z "$PYTHON_EXE" ]; then
    echo "Error: Solo se admite Python 3.12.3. Instala esa versión y agrégala al PATH."
    exit 1
fi
PYTHON_PATH=$($PYTHON_EXE -c "import sys; print(sys.executable)")
echo "Usando Python: $PYTHON_PATH (solo 3.12.3 soportado)"

echo "============================================"
echo "  $PROJECT_NAME Project Setup & Run"
echo "============================================"

# Main execution starts here
# ...existing code for environment setup, extraction, etc...
