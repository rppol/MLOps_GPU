#!/bin/bash
source ~/.bashrc
conda activate mlops
python src/log_production_model.py
cp production_model/xgboost.model ../model_repository/fil/1/
echo "Copied Model"