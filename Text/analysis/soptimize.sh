#!/bin/bash
#SBATCH -A Sis24_laio
#SBATCH -p boost_usr_prod
#SBATCH --job-name=opt
# SBATCH --qos=boost_qos_dbg 
# SBATCH --time 00:30:00
#SBATCH --qos=normal
#SBATCH --time 24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12G
#SBATCH --array=10-30:2 #before was 8-30:2
#SBATCH --output=./log_opt/%x.o%A-%a
#SBATCH --error=./log_opt/%x.o%A-%a

export JAX_ENABLE_X64=True
export JAX_DEBUG_NANS=True
LLM=$1
corpus=$2
randomize=$3
batch_randomize=$4
Nbits=$5
layer_id=$6
alphamax_id=$7
alphamin_id=$8
python3 -u LLM/optimize.py "$LLM" "$corpus" "$randomize" "$batch_randomize" "$Nbits" "$layer_id" "$alphamax_id" "$alphamin_id"