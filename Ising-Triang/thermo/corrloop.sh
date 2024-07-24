#!/bin/bash
L=$1
i0_max=10  #$(($L-1))
j0_max=10 #$(($L-1))
for (( j0=0; j0<$j0_max; j0++ ))
do
  for (( i0=0; i0<=$i0_max; i0++ ))
  do
    echo i0=$i0,j0=$j0
    # sbatch smain.sh $L $i0 $j0
    sleep 0.1
  done
done
