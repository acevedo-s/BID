#!/bin/bash
#SBATCH --job-name=opt0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --partition=regular2,regular1
#SBATCH --array=0-3:1 # for the alpha maxs in parallel
#SBATCH --output=./log_opt/%x.o%A-%a
#SBATCH --error=./log_opt/%x.o%A-%a
export JAX_ENABLE_X64=true
python3 -u optimize.py $1


