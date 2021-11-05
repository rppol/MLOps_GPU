#!/bin/bash
export PATH="/home/jenkins/miniconda/bin:$PATH"
source ~/.bashrc 
conda activate mlops
pip istall dask-ml
pip install sklearn
pip install pandas
python /home/nvidiatest/mlops_blog/src/test_and_evaluate.py