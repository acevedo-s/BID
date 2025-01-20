#!/bin/bash
#SBATCH --job-name=R-FCI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=20G
#SBATCH --partition=regular2,regular1
# SBATCH --qos=fastlane
#SBATCH --output=./log/%x.o%j              # Standard output
#SBATCH --error=./log/%x.o%j               # Standard error

L=$1
T=$2
M_flag=$3
python3 -u compute.py $L $T $M_flag

