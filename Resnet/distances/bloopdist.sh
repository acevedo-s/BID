#!/bin/bash

mkdir -p log_dist

resize=1
export resize

crop_step=28         # because 224 / 8 = 28
mincrop_index=1      # minimum crop index:1
maxcrop_index=1      # maximum crop index:8
layer_ids=({0..7})   # maximum layer index:7 (previous to flatten)
class_ids=({-1..-1})   # maximum class index:6

for layer_id in "${layer_ids[@]}"
do
  for class_id in "${class_ids[@]}"
  do
    for (( crop_index=mincrop_index; crop_index<=maxcrop_index; crop_index++ ))
    do
      crop_size=$((crop_step*crop_index))
      sbatch sdistances.sh $crop_size $class_id $layer_id
      echo crop_size:$crop_size class_id:$class_id layer_id:$layer_id
    done
  done
done