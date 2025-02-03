#!/bin/bash
#SBATCH --job-name=opt-Ns
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH --partition=regular1,regular2
# SBATCH --qos=fastlane # for debugging
#SBATCH --output=./log_opt/%x.o%j              # Standard output
#SBATCH --error=./log_opt/%x.o%j               # Standard error

Ns=$1
T=$2
alphamax=$3
export JAX_ENABLE_X64=True
python3 -u optimize.py $Ns $T $alphamax