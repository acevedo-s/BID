#!/bin/bash

readarray -t T_list < T_list.txt
readarray -t alphamax_list < alphamax_list.txt

for T in "${T_list[@]}"; do
  for alphamax in "${alphamax_list[@]}"; do
    job=$(sbatch soptimize.sh 0.001 $T $alphamax)
    echo $job
    sleep .1
  done
done
