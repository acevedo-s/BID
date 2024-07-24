#!/bin/bash
#SBATCH --job-name=dists0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --partition=regular1,regular2 # long2,long1
#SBATCH --output=./log_dist0/%x.o%j              # Standard output
#SBATCH --error=./log_dist0/%x.o%j               # Standard error
python3 -u dists0.py