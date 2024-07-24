#!/bin/bash
mkdir -p log_dist

Rmin=6
Rmax=10
L=10000
T=2.0
R_scale_factors=(10)

for (( R_id=Rmin; R_id<=Rmax; R_id++ ))
do
  for R_scale_factor in "${R_scale_factors[@]}"
  do
    R=$((R_id * R_scale_factor))
    echo L="$L",R="$R"
    MEMORY=$((R * 3))
    job=$(sbatch --mem="${MEMORY}G" sdistances.sh "$L" "$T" "$R")
    echo "$job"
    sleep .01
  done
done
