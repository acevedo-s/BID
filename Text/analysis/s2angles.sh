#!/bin/bash
#SBATCH -A Sis24_laio
#SBATCH -p boost_usr_prod
#SBATCH --job-name=LLM-angles
# SBATCH --qos=boost_qos_dbg 
# SBATCH --time 00:30:00
#SBATCH --qos=normal
#SBATCH --time 24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --output=./log_angles/%x.o%A-%a
#SBATCH --error=./log_angles/%x.o%A-%a

LLM=$1
corpus=$2
randomize=$3
batch_randomize=$4
Nbits=$5
layer_id=$6
tau=$7

python3 -u LLM/2angles.py "$LLM" "$corpus" "$randomize" "$batch_randomize" "$Nbits" "$layer_id" "$tau"

# for JAX:
# export MPI4JAX_USE_CUDA_MPI=1
# module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8
# mpirun -np $SLURM_NTASKS bash -c "export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK; python3 -u OPT/activations.py $1 $2" # corpus, randomize_flag,