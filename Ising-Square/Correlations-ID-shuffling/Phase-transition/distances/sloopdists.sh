#!/bin/bash

T_idx=0
while read T; do
  if [ "$T_idx" = "0" ]; then
    # FIRST=$(sbatch --parsable sdists.sh $T)
    FIRST=$(sbatch sdists.sh $T)
    echo $FIRST
  else
    # SECOND=$(sbatch --dependency=afterok:$FIRST --parsable sdists.sh $T)
    SECOND=$(sbatch sdists.sh $T)
    echo $SECOND
    # FIRST=$SECOND
  fi
  ((T_idx++))
  sleep .2
done <T_list.txt
echo "T_idx=$T_idx tutto inviato"