#!/bin/bash

mkdir -p log_angles

LLM="Pythia"
corpus="Wikitext"
batch_randomize=0
randomize=0
Nbits=1
l_list=(0) # check that this is a subset of "layer_ids" in parameters...
N_batches_list=(1)
N_batches_list+=({30..60..10})

for N_batches in "${N_batches_list[@]}"
do
  for l in "${l_list[@]}"
  do
    layer_id=$((l))
    echo layer_id:"$layer_id", N_batches:"$N_batches"
    sbatch ssign.sh "$LLM" "$corpus" "$randomize" "$batch_randomize" "$Nbits" "$layer_id" "$N_batches"
    sleep 0.01
  done
done