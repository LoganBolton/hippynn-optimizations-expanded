#!/bin/bash
#SBATCH --job-name=methane_test_report
#SBATCH --output=./logs/submissions/%j_methane_test_report.out
#SBATCH --error=./logs/submissions/%j_methane_test_report.err
#SBATCH --time=00:10:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

METHANE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$METHANE_DIR"
python analysis/report_l3n5_l3n4_test_sets.py
