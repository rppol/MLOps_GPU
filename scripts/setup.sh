#!/bin/bash
source /home/jenkins/miniconda3/etc/profile.d/conda.sh
cd /home/nvidiatest/mlops_blog
conda activate mlops
python src/train.py