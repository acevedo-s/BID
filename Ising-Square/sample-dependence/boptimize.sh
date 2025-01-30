#!/bin/bash

mkdir -p log_opt

T_list=(2.0 2.3 3.0)
alphamax_list=(0.4)

# T_list=(3.0)
Ns_list=({100..1000..100})
# alphamax_list=(0.4)

# T_list=(2.3)
# Ns_list=({2000..5000..1000})
# alphamax_list=(0.2)

for T in "${T_list[@]}"
do
  for Ns in "${Ns_list[@]}"
  do
    for alphamax in "${alphamax_list[@]}"
    do
      echo Ns=$Ns,T=$T,alphamax=$alphamax
      sbatch soptimize.sh $Ns $T $alphamax
    done
  done
done 