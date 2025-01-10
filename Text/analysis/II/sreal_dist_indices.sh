#!/bin/bash
#SBATCH -A Sis24_laio
#SBATCH -p boost_usr_prod
#SBATCH --job-name=LLM-d_indcs
# SBATCH --qos=boost_qos_dbg 
# SBATCH --time 00:30:00
# SBATCH --qos=normal
# SBATCH --time 24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=150G
#SBATCH --output=./log_dists/%x.o%j
#SBATCH --error=./log_dists/%x.o%j

LLM=$1
corpus=$2
layer_id=$3
sub_length=$4
layer_normalize=$5
Ntokens=$6
N_batches=$7

python3 -u real_dist_indices.py \
"$LLM" "$corpus" "$layer_id" "$sub_length" "$layer_normalize" "$Ntokens" "$N_batches"

# for JAX:
# export MPI4JAX_USE_CUDA_MPI=1
# module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8
# mpirun -np $SLURM_NTASKS bash -c "export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK; python3 -u OPT/activations.py $1 $2" # corpus, randomize_flag,