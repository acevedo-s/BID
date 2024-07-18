#!/bin/bash
#SBATCH -A Sis24_laio
#SBATCH -p boost_usr_prod
#SBATCH --job-name=OPT-fig
# SBATCH --qos=normal
# SBATCH --time 2:00:00
#SBATCH --qos=boost_qos_dbg 
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --output=./log_output/%x.o%j              # Standard output
#SBATCH --error=./log_output/%x.o%j               # Standard error

python3 -u LLM/fig-examples.py $1 $2 $3