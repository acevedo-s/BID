#!/bin/bash
#SBATCH --job-name=T-GRIDE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH -A Sis24_laio
#SBATCH -p boost_usr_prod
# SBATCH --qos=boost_qos_dbg
# SBATCH --time 00:30:00
#SBATCH --output=./log/%x.o%j              # Standard output
#SBATCH --error=./log/%x.o%j               # Standard error

LLM=$1
corpus=$2
randomize=$3
batch_randomize=$4
Nbits=$5
layer_id=$6
sub_length=$7

python3 -u compute.py $LLM $corpus $randomize $batch_randomize $Nbits $layer_id $sub_length

