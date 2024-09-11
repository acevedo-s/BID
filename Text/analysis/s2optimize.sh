#!/bin/bash
#SBATCH -A Sis24_laio
#SBATCH -p boost_usr_prod
#SBATCH --job-name=opt
#SBATCH --qos=boost_qos_dbg 
#SBATCH --time 00:30:00
# SBATCH --qos=normal
# SBATCH --time 24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12G
#SBATCH --output=./log_opt/%x.o%j
#SBATCH --error=./log_opt/%x.o%j

export JAX_ENABLE_X64=True
export JAX_DEBUG_NANS=True
randomize=0
Nbits=1

LLM=$1
corpus=$2
batch_randomize=$3
layer_id=$4
alphamax_id=$5
alphamin_id=$6
tau=$7


python3 -u LLM/2optimize.py "$LLM" "$corpus" "$randomize" "$batch_randomize" "$Nbits" "$layer_id" "$alphamax_id" "$alphamin_id" "$tau"