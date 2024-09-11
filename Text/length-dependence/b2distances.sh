#!/bin/bash

mkdir -p log_2distances

LLM=$1
corpus=$2
randomize=0
batch_randomize=$3
Nbits=1
l_list=(24) # check that this is a subset of "layer_ids" in parameters...
tau_list=({9..299..10})
export t=0

for tau in "${tau_list[@]}"
do
  for l in "${l_list[@]}"
  do
    layer_id=$((l))
    echo layer_id:"$layer_id", tau:"$tau"
    sbatch s2distances.sh "$LLM" "$corpus" "$randomize" "$batch_randomize" "$Nbits" "$layer_id" "$tau"
    sleep 0.01
  done
done