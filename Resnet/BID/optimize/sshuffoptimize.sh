#!/bin/bash
#SBATCH --job-name=R-opt
#SBATCH --partition=regular2,regular1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH --array=2-2 # 6 is max index here, for r_id
# SBATCH --qos=fastlane # for debugging
#SBATCH --output=./log_opt/%x.o%A-%a   # Standard output
#SBATCH --error=./log_opt/%x.o%A-%a   # Standard error

export JAX_ENABLE_X64=True
crop_size=$1
layer_id=$2
class_id=$3
alphamin=$4
alphamax=$5

python3 -u shuffoptimize.py "$crop_size" "$layer_id" "$class_id" "$alphamin" "$alphamax"
