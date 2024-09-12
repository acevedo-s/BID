#!/bin/bash

export JAX_ENABLE_X64=True

LLM=$1
corpus=$2
randomize=$3
batch_randomize=$4
Nbits=$5

python3 -u LLM/_utils/scale_selection.py "$LLM" "$corpus" "$randomize" "$batch_randomize" "$Nbits"