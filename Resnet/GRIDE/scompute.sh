#!/bin/bash
#SBATCH --job-name=R-gride
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=120G
#SBATCH --partition=regular2,regular1
#SBATCH --output=./log_gride/%x.o%j              # Standard output
#SBATCH --error=./log_gride/%x.o%j               # Standard error

crop_size=$1
class_id=$2
layer_id=$3
dbg=$4
shuffle=$5

if [ "$shuffle" -eq 0 ]; then
  python3 -u compute.py $crop_size $class_id $layer_id $dbg
else
  python3 -u shuff_compute.py $crop_size $class_id $layer_id $dbg
fi



