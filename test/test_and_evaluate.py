import sys
sys.path.append('src/')

import os
import argparse

import numpy as np
import xgboost as xgb
from read_params import read_params
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from dask_client import dask_client
from load_data import load_data
from feature_engg import feature_engg

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return (rmse, mae, r2)

def test_and_evaluate(client, dtest, config_path, model=None):
    config = read_params(config_path)

    #Load Model
    if model == None:
        model = xgb.Booster()
        saved_model_dir = 'production_model'
        model_name = config["test"]["model_name"]
        model_extension = config["test"]["model_extension"]
        model.load_model(os.path.join(saved_model_dir, model_name+model_extension))

    #Drop Actual value
    actual = dtest[config["base"]["target_col"]].compute().to_array()
    dtest = dtest.drop([config["base"]["target_col"]], axis=1)

    #Make Prediction
    dtest = xgb.dask.DaskDMatrix(client, dtest)
    pred = xgb.dask.predict(client, model, dtest).compute()

    #Evaluate Prediction
    rmse, mae, r2 = eval_metrics(actual, pred)
    print("Root Mean Squared Error : ", rmse)
    print("Mean Absolute Error : ", mae)
    print("R-squared Score : ", r2)
    return (rmse, mae, r2)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    client = dask_client(config_path=parsed_args.config)
    df = load_data(config_path=parsed_args.config, test=True)
    df = feature_engg(df)
    _1, _2, _3 = test_and_evaluate(client, df, config_path=parsed_args.config)
    client.close()