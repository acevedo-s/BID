#!/bin/bash

LLM=$1
corpus=$2
randomize=$3
batch_randomize=$4
Nbits=$5
FIRST=$6

export JAX_PLATFORMS=cpu
if [ -z "$FIRST" ]; then
  dependency=""
else
  dependency="--dependency=afterok:$FIRST"
  echo "$dependency"
fi

l_list=(24 0) # check that this is a subset of "layer_ids" in parameters...

for l in "${l_list[@]}"
do
  layer_id=$((l))
  echo "layer_id: $layer_id"
  sbatch $dependency sdistances.sh "$LLM" "$corpus" "$randomize" "$batch_randomize" "$Nbits" "$layer_id"
done