#!/bin/bash
set -euo pipefail

# Fast syntax check
uv run python -c "import ast; ast.parse(open('train_model.py').read()); print('Syntax OK')" > /dev/null 2>&1

# Run the model training and evaluation
uv run python train_model.py
