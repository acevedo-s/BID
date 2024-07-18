#!/bin/bash

export JAX_ENABLE_X64=True

LLM=$1
corpus=$2
randomize=$3
# layer_id=$4
python3 -u LLM/_utils/rhists.py "$LLM" "$corpus" "$randomize" #$layer_id