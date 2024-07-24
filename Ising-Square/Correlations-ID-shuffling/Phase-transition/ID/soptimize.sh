#!/bin/bash
#SBATCH --job-name=opt-shuff
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --partition=regular2,regular1
#SBATCH --array=1-200:1
#SBATCH --output=./log_opt/%x.o%A-%a
#SBATCH --error=./log_opt/%x.o%A-%a
export JAX_ENABLE_X64=true
python3 -u optimize.py $1 $2 $3 #alphamin,T,alphamax


