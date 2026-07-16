#!/bin/bash
set -e

METHANE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$METHANE_DIR/../.." && pwd)"
ENV_NAME="${ENV_NAME:-hippynn-expanded-lmax}"

echo "Creating conda environment '$ENV_NAME' from configs/environment.yml..."
cd "$REPO_ROOT"
conda env create -f "$METHANE_DIR/configs/environment.yml" -n "$ENV_NAME"

echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "Installing this checkout of hippynn in editable mode..."
python -m pip install -e "$REPO_ROOT"

echo "Testing installation..."
python -c "import sys; print('Python:', sys.version)"
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import ase; print('ASE version:', ase.__version__)"
python -c "import hippynn; print('Hippynn installed at:', hippynn.__file__)"

echo ""
echo "Environment '$ENV_NAME' created successfully!"
echo "To activate: conda activate $ENV_NAME"
