#!/bin/bash

mkdir -p log_opt

layer_ids=({7..7})   # maximum layer index:7 (previous to flatten)
mincrop_index=4      # minimum crop index:1
maxcrop_index=8      # maximum crop index:8
class_ids=({-1..-1})   # maximum class index:6
alphamins=(0.01)
alphamaxs=(0.1 0.2 0.3 0.4 0.5)

for layer_id in "${layer_ids[@]}"
do
  for class_id in "${class_ids[@]}"
  do
    for alphamax in "${alphamaxs[@]}"
    do
      for alphamin in "${alphamins[@]}"
      do
        for (( crop_index=mincrop_index; crop_index<=maxcrop_index; crop_index++ ))
        do
          crop_size=$((28*crop_index))
          if [ "$class_id" -eq -1 ]; then
              sbatch sshuffoptimize.sh "$crop_size" "$layer_id" "$class_id" "$alphamin" "$alphamax"
          else
              sbatch soptimize.sh "$crop_size" "$layer_id" "$class_id" "$alphamin" "$alphamax"
          fi
          echo crop_size:$crop_size,layer_id:$layer_id,class_id:$class_id,alphamax:$alphamax,alphamin:$alphamin
          sleep .01
        done
      done
    done
  done
done