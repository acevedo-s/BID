#!/bin/bash
#SBATCH --job-name=Isng-dist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH --partition=regular2,regular1
#SBATCH --array=0-39
# SBATCH --qos=fastlane # for debugging
#SBATCH --output=./log_dist/%x.o%A-%a   # Standard output
#SBATCH --error=./log_dist/%x.o%A-%a   # Standard error

L=$1
python3 -u distances.py $L