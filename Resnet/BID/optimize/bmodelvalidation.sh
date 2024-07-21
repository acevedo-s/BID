#!/bin/bash
export JAX_ENABLE_X64=True
python model_validation.py $1 1 #crop_size,plot_fit