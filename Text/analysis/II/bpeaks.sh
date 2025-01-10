#!/bin/bash

mkdir -p log_peaks

LLM=OPT #Pythia
corpus=OWebtext #Wikitext
normalize_flags=(0)
# export N_batches=50
# layer_ids=({0..24})
sub_lengths=(20 200 300)
layer_ids=(0 1 12 18 24)
# sub_lengths=(100 300)
# sub_lengths=({20..300..20})
sample_idcs=(1 2 3)

for layer_normalize in "${normalize_flags[@]}"
do
  for layer_id in "${layer_ids[@]}"
  do
    for sub_length in "${sub_lengths[@]}"
    do
      # Ntokens=$((sub_length-1))
      Ntokens=0
      for sample_idx0 in "${sample_idcs[@]}"
      do
        echo layer_id="$layer_id" sub_length="$sub_length" \
        Ntokens="$Ntokens" sample_idx0="$sample_idx0"

        sbatch speaks.sh "$LLM" "$corpus" \
        "$layer_id" "$sub_length" "$layer_normalize" "$Ntokens" "$sample_idx0"
        sleep .005
      done
    done
  done
done