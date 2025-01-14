#!/bin/bash

# T_list=($(seq 1.8 0.1 3))
# T_list+=($(seq 2.21 0.01 2.39))
T_list=(1.7 1.8 2.3 3 4)
L_list=({30..100..10})
M_flags=(1)
# L_list=(30)
global_flags=(1)

for global_flag in "${global_flags[@]}"
do
  for M_flag in "${M_flags[@]}"
  do
    for T in "${T_list[@]}"
    do
      for L in "${L_list[@]}"
      do
        echo L=$L,T=$T,M_flag=$M_flag
        sbatch scompute.sh $L $T $M_flag $global_flag
      done
    done 
  done
done