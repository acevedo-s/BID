#!/bin/bash
#SBATCH -A Sis24_laio
#SBATCH -p boost_usr_prod
#SBATCH --job-name=RandomTokensGen
#SBATCH --qos=boost_qos_dbg 
#SBATCH --time 00:30:00
# SBATCH -p boost_usr_prod
# SBATCH --time 3:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --output=./log_output/%x.o%j              # Standard output
#SBATCH --error=./log_output/%x.o%j               # Standard error

LLM=$1
corpus=RandomTokens
randomize=0

python3 -u LLM/random_token_generator.py $1 $corpus $randomize