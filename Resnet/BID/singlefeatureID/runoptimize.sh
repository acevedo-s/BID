#!/bin/bash
#SBATCH --job-name=KL-opt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --partition=long2,long1 # regular2,regular1
#SBATCH --output=./log_output/%x.o%j   # Standard output
#SBATCH --error=./log_output/%x.o%j    # Standard error

python3 optimize.py