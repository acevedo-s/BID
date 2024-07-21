#!/bin/bash
#SBATCH --job-name=R-opt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH --partition=regular2,regular1
#SBATCH --array=0-6 # for classes
# SBATCH --qos=fastlane # for debugging
#SBATCH --output=./log_output/%x.o%A-%a   # Standard output
#SBATCH --error=./log_output/%x.o%A-%a   # Standard error

export JAX_ENABLE_X64=True
python3 -u classoptimize.py $1 $2 # crop_size, layer_id