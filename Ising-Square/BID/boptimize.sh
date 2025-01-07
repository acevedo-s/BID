#!/bin/bash

min_seed=1
max_seed=1

file_path="T_list.txt"
L_list=(40 50 60 70 80 90)
alphamax_list=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45)
# alphamax_list=(0.1)
while IFS= read -r T || [ -n "$T" ]
do
  for L in "${L_list[@]}"
  do
    for alphamax in "${alphamax_list[@]}"
    do
      echo L=$L,T=$T,alphamax=$alphamax
      job=$(sbatch soptimize.sh $L $T $alphamax)
      echo $job
      sleep .01
    done
  done
done < "$file_path"