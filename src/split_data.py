import os
import argparse
from dask_ml.model_selection import train_test_split
from load_data import load_train_data
from feature_engg import feature_engg
from dask_client import read_params
import xgboost as xgb

def split_data(client, df, config_path):
    config = read_params(config_path)
    split_ratio = config["split_data"]["valid_test_ratio"]
    random_state = config["base"]["random_state"]
    y = df[config["base"]["target_col"]]
    X = df.drop([config["base"]["target_col"]], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=random_state)
    dtrain, dvalid = generate_Dmatrix(client, X_train, y_train, X_test, y_test)
    return (client, dtrain, dvalid)

def generate_Dmatrix(client, X_train, y_train, X_test, y_test):
    dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)
    dvalid = xgb.dask.DaskDMatrix(client, X_test, y_test)
    return (dtrain, dvalid)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    client, df = load_train_data(config_path=parsed_args.config)
    client, df = feature_engg(client, df)
    split_data(client, df,config_path=parsed_args.config)