#!/bin/bash
#SBATCH --job-name=act-Resnet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH --partition=regular2,regular1
# SBATCH --qos=fastlane # for debugging
#SBATCH --output=./log_a/%x.o%j              # Standard output
#SBATCH --error=./log_a/%x.o%j               # Standard error

crop_size=$1
class_id=$2
python3 -u compute_activations.py $crop_size $class_id