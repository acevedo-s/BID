#!/bin/bash

mkdir -p log_angles

LLM=$1
corpus=$2
randomize=0 #$3
batch_randomize=1 #$4
Nbits=0
l_list=(0) # check that this is a subset of "layer_ids" in parameters...
tau_list=(0)
tau_list+=({9..299..10})
export t=0

for tau in "${tau_list[@]}"
do
  for l in "${l_list[@]}"
  do
    layer_id=$((l))
    echo layer_id:"$layer_id", tau:"$tau"
    sbatch s2angles.sh "$LLM" "$corpus" "$randomize" "$batch_randomize" "$Nbits" "$layer_id" "$tau"
    sleep 0.01
  done
done