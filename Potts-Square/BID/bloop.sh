#!/bin/bash

l_list=(10 9 8 7 6 5 4)
# T_list=(0.5 0.6 0.75 1)
mapfile -t T_list < "T_list37.txt"
alphamax_list=(0.1 0.2)
for T in "${T_list[@]}"
do
  for l in "${l_list[@]}"
  do
    for alphamax in "${alphamax_list[@]}"
    do
      L=$((10*$l))
      echo L=$L,T=$T,alphamax=$alphamax
      sbatch soptimize.sh $L $T $alphamax
      sleep 0.01
    done
  done
done
