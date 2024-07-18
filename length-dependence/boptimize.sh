#!/bin/bash

LLM=$1
corpus=$2
randomize=$3
batch_randomize=$4
Nbits=$5
alphamax_id_max=12 #14
alphamin_id_max=2 #4
l_list=(0 24)
# for (( l=0; l<=24; l++ )) ### layer _id
for l in "${l_list[@]}"
  do
  # layer_id=$((l*12))
  layer_id=$l
  for ((alphamax_id=0; alphamax_id<alphamax_id_max; alphamax_id++))
  do
    for ((alphamin_id=0; alphamin_id<alphamin_id_max; alphamin_id++))
    do
      job=$(sbatch soptimize.sh "$LLM" "$corpus" "$randomize" "$batch_randomize" "$Nbits" "$layer_id" $alphamax_id $alphamin_id)
      echo "$job"
      echo alphamax_id:$alphamax_id, alphamin_id:$alphamin_id, layer_id:"$layer_id"
      sleep .05
    done
  done
done
