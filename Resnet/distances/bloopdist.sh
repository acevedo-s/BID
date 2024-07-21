#!/bin/bash

crop_step=28
resize=0
for (( i=1; i<=8; i++ ))
do
  # crop_size=$((14+28*$i))
  crop_size=$(($crop_step*$i))
  job=$(sbatch sdistances.sh $crop_size $resize)
  # job=$(sbatch shuffsdistances.sh $crop_size $resize)
  echo $job
  echo crop_size:$crop_size
  sleep .2
done
