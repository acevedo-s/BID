#!/bin/bash
#SBATCH --job-name=R-dist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH --partition=regular2,regular1
#SBATCH --array=0-8 # for layers
# SBATCH --qos=fastlane # for debugging
#SBATCH --output=./log_dist/%x.o%j              # Standard output
#SBATCH --error=./log_dist/%x.o%j               # Standard error

crop_size=$1
resize=$2
python3 -u distances.py $crop_size $resize