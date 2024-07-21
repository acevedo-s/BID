#!/bin/bash
#SBATCH --job-name=distances
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH --partition=regular2,regular1
#SBATCH --output=./log_output/%x.o%j   # Standard output
#SBATCH --error=./log_output/%x.o%j    # Standard error

python3 computedistances.py