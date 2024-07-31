#!/bin/bash

mkdir -p log_opt

# layer_ids=({2..7})   # maximum layer index:7 (previous to flatten)
# mincrop_index=1      # minimum crop index:1
# maxcrop_index=8      # maximum crop index:8
# class_ids=({2..6})   # maximum class index:6
# alphamins=(0.01)
# alphamaxs=(0.1 0.2 0.3 0.4 0.5)

layer_ids=({0..1})   # maximum layer index:7 (previous to flatten)
mincrop_index=1      # minimum crop index:1
maxcrop_index=8      # maximum crop index:8
class_ids=({2..6})   # maximum class index:6
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
          sbatch sclassoptimize.sh $crop_size $layer_id $class_id $alphamin $alphamax
          echo crop_size:$crop_size,layer_id:$layer_id,class_id:$class_id,alphamax:$alphamax,alphamin:$alphamin
          sleep .01
        done
      done
    done
  done
done