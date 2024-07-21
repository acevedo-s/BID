#!/bin/bash

# for i in {2..8}
# for (( i=9; i>=1; i-- ))
layer_id=$1
for (( i=1; i<=3; i++ ))
do
  crop_size=$((28*$i))
  job=$(sbatch sclassoptimize.sh $crop_size $layer_id)
  echo crop_size:$crop_size
  echo $job
  sleep .2
done
