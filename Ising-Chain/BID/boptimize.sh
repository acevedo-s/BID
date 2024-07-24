#!/bin/bash

Rmin=6
Rmax=10
L=10000
T=2
# alphamax_list=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45)
alphamax_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)
# alphamax_list=(0.2 0.3 0.4 0.6 0.7 0.8 0.9)
scale_factors=(10)
  for (( R_id=Rmin; R_id<=Rmax; R_id++ ))
  do
    for scale_factor in "${scale_factors[@]}"
    do
      R=$((R_id * scale_factor))
        for alphamax in "${alphamax_list[@]}"
        do
          echo R=$R,alphamax="$alphamax"
          sbatch soptimize.sh $L "$T" "$R" "$alphamax" #| awk '{print job_id:$4}'
          sleep .01
        done
      done
    done
