#!/bin/bash
#SBATCH --job-name=shuff-dists
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --partition=regular1,regular2
#SBATCH --array=1-100:1
#SBATCH --output=./log_dist/%x.o%A-%a
#SBATCH --error=./log_dist/%x.o%A-%a

python3 -u dists.py $1