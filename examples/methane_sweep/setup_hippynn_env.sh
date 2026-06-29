#!/bin/bash
set -e

echo "Creating conda environment 'hippynn'..."
conda create -n hippynn python=3.12 -y

echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hippynn

echo "Installing PyTorch..."
conda install -y -c pytorch -c nvidia \
  pytorch torchvision torchaudio pytorch-cuda=11.8 \
  "mkl<2025" "intel-openmp<2025"

echo "Installing hippynn..."
cd /vast/home/logan_bolton/Github/hippynn-optimizations-expanded
python -m pip install -e .

echo "Installing additional dependencies..."
python -m pip install wandb ase

echo "Testing installation..."
python -c "import sys; print('Python:', sys.version)"
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import ase; print('ASE version:', ase.__version__)"
python -c "import hippynn; print('Hippynn installed at:', hippynn.__file__)"

echo ""
echo "Environment 'hippynn' created successfully!"
echo "To activate: conda activate hippynn"