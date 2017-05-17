#!/usr/bin/env bash
rm *.out
rm *.tsv
export CUDA_VISIBLE_DEVICES=1; python3 -u model_run.py m3 > m3.out
export CUDA_VISIBLE_DEVICES=1; python3 -u model_run.py m5 > m5.out
export CUDA_VISIBLE_DEVICES=1; python3 -u model_run.py m11 > m11.out
export CUDA_VISIBLE_DEVICES=1; python3 -u model_run.py m18 > m18.out
export CUDA_VISIBLE_DEVICES=1; python3 -u model_run.py m34 > m34.out
