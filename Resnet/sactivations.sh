#!/bin/bash
#SBATCH --job-name=act-resnet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH --partition=regular2,regular1
#SBATCH --output=./log_output/%x.o%j              # Standard output
#SBATCH --error=./log_output/%x.o%j               # Standard error

crop_size=$1
resize=$2
python3 -u compute_activations.py $crop_size $resize