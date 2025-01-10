#!/bin/bash
#SBATCH -A Sis24_laio
#SBATCH -p boost_usr_prod
#SBATCH --job-name=weights
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
# SBATCH --qos=normal
# SBATCH --time 24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --output=./log_e/%x_%j.o
#SBATCH --error=./log_e/%x_%j.o

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

f=$1
layer_idx=$2

python3 -u extract.py "$f" "$layer_idx"

# for JAX:
# export MPI4JAX_USE_CUDA_MPI=1
# module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8
# mpirun -np $SLURM_NTASKS bash -c "export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK; python3 -u OPT/activations.py $1 $2" # corpus, randomize_flag,