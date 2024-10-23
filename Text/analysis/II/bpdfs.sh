#!/bin/bash

mkdir -p log_peaks

LLM=OPT #Pythia
corpus=OWebtext #Wikitext
normalize_flags=(0)
# layer_ids=({0..24})
sub_lengths=(20)
layer_ids=(0 18 24)
# sub_lengths=(100 300)
# sub_lengths=({20..300..20})
Ntokens=0
export N_batches=50

for layer_normalize in "${normalize_flags[@]}"
do
  for layer_id in "${layer_ids[@]}"
  do
    for sub_length in "${sub_lengths[@]}"
    do
    # Ntokens=$((sub_length-1))
      echo layer_id="$layer_id" sub_length="$sub_length" N_batches="$N_batches"

      sbatch scdfs.sh "$LLM" "$corpus" \
      "$layer_id" "$sub_length" "$layer_normalize" "$Ntokens"
      sleep .005
    done
  done
done