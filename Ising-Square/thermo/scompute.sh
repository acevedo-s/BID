#!/bin/bash
#SBATCH --job-name=thermo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=30G
#SBATCH --partition=regular2,regular1
# SBATCH --qos=fastlane
#SBATCH --output=./log_thermo/%x.o%j              # Standard output
#SBATCH --error=./log_thermo/%x.o%j               # Standard error

python3 -u compute.py 

