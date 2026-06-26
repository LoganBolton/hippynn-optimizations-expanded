#!/bin/bash
set -e

echo "Creating conda environment 'hippynn'..."
conda create -n hippynn python=3.10 -y

echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate hippynn

echo "Installing PyTorch..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

echo "Installing hippynn..."
cd /vast/home/logan_bolton/Github/hippynn-optimizations-expanded
pip install -e .

echo "Installing additional dependencies..."
pip install wandb numpy

echo "Testing installation..."
python -c "import hippynn; print('Hippynn installed at:', hippynn.__file__)"
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo ""
echo "✅ Environment 'hippynn' created successfully!"
echo "To activate: conda activate hippynn"
