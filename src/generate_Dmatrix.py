import argparse
import xgboost as xgb

from dask_client import dask_client
from load_data import load_data
from feature_engg import feature_engg
from split_data import split_data

def generate_Dmatrix(client, X_train, X_test, y_train, y_test):
    dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)
    dvalid = xgb.dask.DaskDMatrix(client, X_test, y_test)
    return (dtrain, dvalid)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    client = dask_client(config_path=parsed_args.config)
    df = load_data(config_path=parsed_args.config)
    df = feature_engg(df)
    X_train, X_test, y_train, y_test = split_data(df, config_path=parsed_args.config)
    _1, _2 = generate_Dmatrix(client, X_train, X_test, y_train, y_test)
    client.close()