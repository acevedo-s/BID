#!/bin/bash
#SBATCH --job-name=half-dist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH --partition=regular2,regular1
# SBATCH --array=0-48 # 49 temperatures from T=1 to T=4
# SBATCH --qos=fastlane # for debugging
#SBATCH --output=./log_dist/%x.o%j   # Standard output
#SBATCH --error=./log_dist/%x.o%j    # Standard error

L=$1
T=$2
half=$3
python3 -u halfdistances.py $L $T $half