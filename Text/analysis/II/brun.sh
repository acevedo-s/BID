#!/bin/bash

mkdir -p log_dists
mkdir -p log_ranks

LLM=Pythia #$1
corpus=Wikitext #$2
layer_ids=({0..24})
sub_lengths=(10 50 200)
# layer_ids=(0 1 19 24)
# sub_lengths=({20..300..20})

debuggin=0
normalize_flags=(0)

if [ $debuggin == 1 ]; then
  qos=boost_qos_dbg
  time=00:30:00
else
  qos=normal
  time=24:00:00
fi

for layer_normalize in "${normalize_flags[@]}"
do
  for layer_id in "${layer_ids[@]}"
    do
    for sub_length in "${sub_lengths[@]}"
    do

      JOB_ID1=$(sbatch --qos="$qos" --time="$time" \
      sreal_dist_indices.sh \
      "$LLM" "$corpus" "$layer_id" "$sub_length" "$layer_normalize" \
      | awk '{print $4}')

      JOB_ID2=$(sbatch --qos="$qos" --time="$time" \
      sspin_distances.sh \
      "$LLM" "$corpus" "$layer_id" "$sub_length" "$layer_normalize" \
      | awk '{print $4}')

      sbatch --dependency=afterok:$JOB_ID1:$JOB_ID2 \
      sRS_ranks.sh "$LLM" "$corpus" "$layer_id" "$sub_length" "$layer_normalize"
      sbatch --dependency=afterok:$JOB_ID1:$JOB_ID2 \
      sSR_ranks.sh "$LLM" "$corpus" "$layer_id" "$sub_length" "$layer_normalize"

      echo layer_id="$layer_id" sub_length="$sub_length"
      sleep .005
    done
  done
done