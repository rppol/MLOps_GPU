#!/bin/bash
source ~/.bashrc
conda activate mlops
python src/log_production_model.py
yes | cp -rf production_model/xgboost.model ../model_repository/fil/1/