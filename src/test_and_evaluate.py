import os
import argparse
from dask_client import read_params
from load_data import load_test_data
from feature_engg import feature_engg
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

def eval_metrics(actual, pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def test_and_evaluate(client, df, config_path):
    config = read_params(config_path)

    #Load Model
    model = xgb.Booster()
    model_name = config["test"]["model_name"]
    model_extension = config["test"]["model_extension"]
    model.load_model(os.path.join("saved_models", model_name+model_extension))

    #Drop True value
    actual = df[config["base"]["target_col"]]
    df = df.drop([config["base"]["target_col"]], axis=1)

    #Make Prediction
    dtest = xgb.dask.DaskDMatrix(client, df)
    prediction = xgb.dask.predict(client, model, dtest)
    pred = prediction.compute()

    #Evaluate Prediction
    actual = actual.compute().to_array()
    print(eval_metrics(actual, pred))

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    client, df = load_test_data(config_path=parsed_args.config)
    client, df = feature_engg(client, df)
    test_and_evaluate(client, df, config_path=parsed_args.config)