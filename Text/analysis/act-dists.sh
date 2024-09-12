#!/bin/bash
#SBATCH -A Sis24_laio
#SBATCH -p boost_usr_prod

LLM=$1
corpus=$2
randomize=$3
batch_randomize=$4

FIRST=$(sbatch --parsable sgactivations.sh "$LLM" "$corpus" "$randomize" "$batch_randomize")
sleep 1
./bdistances.sh "$LLM" "$corpus" "$randomize" "$batch_randomize" "$FIRST"