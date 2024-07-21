#!/bin/bash

for ((layer_id=0; layer_id<=8; layer_id++))
do
  for (( i=1; i<=3; i++ ))
  do
    crop_size=$((28*$i))
    job=$(sbatch sshuffoptimize.sh $crop_size $layer_id)
    echo crop_size:$crop_size $layer_id
    echo $job
    sleep .2
  done
done
