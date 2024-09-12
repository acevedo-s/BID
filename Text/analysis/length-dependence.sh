#!/bin/bash

LLM=$1
corpus=$2
randomize=$3
batch_randomize=$4
Nbits=$5

export JAX_ENABLE_X64=True
python3 LLM/_utils/length-dependence.py "$LLM" "$corpus" "$randomize" "$batch_randomize" "$Nbits"