#!/bin/bash

# min_seed=1
# max_seed=1

# T_list=("1.8" "1.9" "2.0" "2.29" "2.3" "3.00" "4.00")
# for (( seed=min_seed; seed<=max_seed; seed++ ))
# do
#   for T in "${T_list[@]}"
#   do
#     echo seed=$seed,T=$T
#     job=$(sbatch soptimize.sh $T $seed)
#     echo $job
#     sleep .02
#   done
# done


file_path="../distances/T_list40.txt"
L_list=(130)
alphamax_list=(0.1)
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