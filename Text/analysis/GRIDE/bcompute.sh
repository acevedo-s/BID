#!/bin/bash

mkdir -p log

LLM='Pythia'
corpus='Wikitext'
randomize=0
batch_r_flags=(0 1)
Nbits=1
layer_ids=(0 24)
sub_lengths=({20..300..20})
export N_batches=50

for batch_randomize in "${batch_r_flags[@]}"
do
  for layer_id in "${layer_ids[@]}"
  do
    for sub_length in "${sub_lengths[@]}"
    do
      echo layer_id=$layer_id sub_length=$sub_length batch_randomize=$batch_randomize
      sbatch  scompute.sh $LLM $corpus $randomize $batch_randomize $Nbits $layer_id $sub_length
    done
  done 
done
