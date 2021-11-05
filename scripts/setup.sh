#!/bin/bash
export PATH="/home/jenkins/miniconda/bin:$PATH"
source ~/.bashrc 
conda activate mlops
pip install dask-ml
python /home/nvidiatest/mlops_blog/src/test_and_evaluate.py