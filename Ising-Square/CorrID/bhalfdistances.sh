#!/bin/bash

L_list=(30 40 50 60 70 80 90 100)
half=1
mapfile -t T_list < "T_list.txt"
for T in "${T_list[@]}"
  do
  for L in "${L_list[@]}"
  do
    echo "L:$L,T:$T"
    job=$(sbatch shalfdistances.sh $L $T $half)
    echo $job
  done
done