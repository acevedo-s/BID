#!/bin/bash

LLM=$1
corpus=$2
randomize=$3
batch_randomize=$4
layer_id=$5


# export JAX_ENABLE_X64=True
python3 LLM/_utils/hists-activations.py "$LLM" "$corpus" "$randomize" "$batch_randomize" "$layer_id"