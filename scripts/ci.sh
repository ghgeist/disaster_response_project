#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$REPO_ROOT/.venv"
PYTHON_BIN="$VENV_PATH/bin/python"

cd "$REPO_ROOT"

echo "Python on PATH:"
which python
python -V

if [[ ! -d "$VENV_PATH" ]]; then
  python -m venv "$VENV_PATH"
fi

"$PYTHON_BIN" -m pip install -U pip

if [[ -f "$REPO_ROOT/pyproject.toml" ]] && rg -q "\[project\.optional-dependencies\]" "$REPO_ROOT/pyproject.toml"; then
  "$PYTHON_BIN" -m pip install -e ".[dev]"
else
  "$PYTHON_BIN" -m pip install -r requirements.txt
  if [[ -f "$REPO_ROOT/requirements-dev.txt" ]]; then
    "$PYTHON_BIN" -m pip install -r requirements-dev.txt
  fi
fi

"$PYTHON_BIN" -c "import sys; print(sys.executable); print(sys.version)"
"$PYTHON_BIN" -c "import pytest; print(pytest.__version__)"

"$PYTHON_BIN" -m pytest -q
