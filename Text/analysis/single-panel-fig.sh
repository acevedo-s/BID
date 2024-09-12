#!/bin/bash

LLM=$1
corpus=$2
alphamin=$3
Nbits=$4
python3 LLM/_utils/single-panel-text-fig.py "$LLM" "$corpus" "$alphamin" "$Nbits"