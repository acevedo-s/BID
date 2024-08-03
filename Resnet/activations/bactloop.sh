#!/bin/bash

crop_step=28          # full image size: 224*224. note that 28 = 224 / 8 
mincrop_index=1
maxcrop_index=8
class_ids=({-1..-1})  # -1 for shuffled

mkdir -p log_a

for class_id in "${class_ids[@]}"
do
  for (( crop_index=mincrop_index; crop_index<=maxcrop_index; crop_index++ ))
  do
    crop_size=$((crop_step*crop_index))
    echo crop_size:$crop_size, class_id:$class_id
    sbatch sactivations.sh $crop_size $class_id
    sleep .01
  done
done