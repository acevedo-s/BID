#!/bin/bash

LLM=$1
corpus=$2
batch_randomize=$3
alphamax_id_max=0
alphamin_id_max=0
l_list=(24)
tau_list=({9..299..10})
export t=0

for layer_id in "${l_list[@]}"
  do
  for tau in "${tau_list[@]}"
  do
    for ((alphamax_id=0; alphamax_id<=alphamax_id_max; alphamax_id++))
    do
      for ((alphamin_id=0; alphamin_id<=alphamin_id_max; alphamin_id++))
      do
        sbatch s2optimize.sh "$LLM" "$corpus" "$batch_randomize" "$layer_id" $alphamax_id $alphamin_id "$tau"
        echo alphamax_id:$alphamax_id, alphamin_id:$alphamin_id, layer_id:"$layer_id", tau:"$tau"
        sleep .01
      done
    done
  done
done