#!/bin/bash

mkdir -p log_gride

dbg=$1
shuffle=0

crop_step=28           # because 224 / 8 = 28
mincrop_index=1        # minimum crop index:1
maxcrop_index=8        # maximum crop index:8
layer_ids=(7)     # maximum layer index:7 (previous to flatten)
class_ids=({0..6})     # maximum class index:6

if [ "$dbg" -eq 1 ]; then
  qos="fastlane"
  layer_ids=({0..0})
  class_ids=({0..0})
  maxcrop_index=1        
else
  qos="normal"
fi

resize=1
export resize

for layer_id in "${layer_ids[@]}"
do
  for class_id in "${class_ids[@]}"
  do
    for (( crop_index=mincrop_index; crop_index<=maxcrop_index; crop_index++ ))
    do
      crop_size=$((crop_step*crop_index))
      sbatch scompute.sh $crop_size $class_id $layer_id $dbg $shuffle
      echo crop_size:$crop_size class_id:$class_id layer_id:$layer_id dbg:$dbg shuffle:$shuffle
    done
  done
done