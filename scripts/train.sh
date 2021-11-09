#!/bin/bash
source ~/.bashrc
conda activate mlops
pip install mlflow
python src/train_with_tracking.py