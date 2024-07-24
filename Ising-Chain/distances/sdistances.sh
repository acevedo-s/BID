#!/bin/bash
#SBATCH --job-name=Ising-dist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
# SBATCH --mem=56G
#SBATCH --partition=regular2,regular1
#SBATCH --output=./log_dist/%x.o%j
#SBATCH --error=./log_dist/%x.o%j
# SBATCH --qos=fastlane # for debugging

# SBATCH --array=0-48 # 49 temperatures from T=1 to T=4
# SBATCH --output=./log_dist/%x.o%A-%a   # Standard output
# SBATCH --error=./log_dist/%x.o%A-%a   # Standard error

L=$1
T=$2
R=$3
python3 -u distances.py "$L" "$T" "$R"