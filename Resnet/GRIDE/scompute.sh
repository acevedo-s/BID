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

if [ "$class_id" -eq -1 ]; then
    python3 -u shuffdistances.py $crop_size $class_id $layer_id
else
    python3 -u compute.py $crop_size $class_id $layer_id $dbg
fi

