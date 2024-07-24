#!/bin/bash

export JAX_ENABLE_X64=True
scale_id=$1
python3 model_validationN.py $scale_id