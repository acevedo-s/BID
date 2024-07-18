#!/bin/bash
#SBATCH -A Sis24_laio
#SBATCH --job-name=tokenization
# SBATCH --qos=boost_qos_dbg 
# SBATCH --time 00:30:00
#SBATCH -p boost_usr_prod
#SBATCH --time 3:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=120G
#SBATCH --output=./log_output/%x.o%j              # Standard output
#SBATCH --error=./log_output/%x.o%j               # Standard error

LLM=$1
corpus=$2
randomize=$3

python3 -u LLM/tokenizer.py $LLM $corpus $randomize