import os
import argparse
import xgboost as xgb
import xgboost as xgb
import numpy as np
import mlflow
import mlflow.xgboost
from read_params import read_params
from urllib.parse import urlparse

from dask_client import dask_client
from load_data import load_data
from feature_engg import feature_engg
from split_data import split_data
from generate_Dmatrix import generate_Dmatrix
from test_and_evaluate import test_and_evaluate

#MLflow server script
"""mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 -p 8003"""

def train_with_tracking(client, dtrain, dvalid, dtest, config_path): 
    config = read_params(config_path)

    remote_server_uri = config["mlflow"]["remote_server_uri"]
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=config["mlflow"]["run_name"]):
        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        params = {
            'booster' : config["train"]["params"]["booster"],
            'eval_metric': config["train"]["params"]["eval_metric"],
            'tree_method':config["train"]["params"]["tree_method"],
            'objective': config["train"]["params"]["objective"],
            'min_child_weight': config["train"]["params"]["min_child_weight"],
            'colsample_bytree': config["train"]["params"]["colsample_bytree"],
            'learning_rate': config["train"]["params"]["learning_rate"]
        }

        num_boost_round = config["train"]["num_boost_round"]
        early_stopping_rounds = config["train"]["early_stopping_rounds"]
        verbose_eval = config["train"]["verbose_eval"]

        model = xgb.dask.train(client, params, dtrain,
                                num_boost_round=num_boost_round,
                                evals=watchlist,
                                early_stopping_rounds=early_stopping_rounds,
                                verbose_eval=verbose_eval)

        rmse, mae, r2 = test_and_evaluate(client, dtest, config_path, model)

        for key, value in params.items():
            mlflow.log_param(key, value)
        
        mlflow.log_param("num_boost_round", num_boost_round)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.xgboost.log_model(model["booster"], "model", registered_model_name=config["mlflow"]["registered_model_name"])
        else:
            mlflow.xgboost.log_model(model["booster"], "model")
    return model

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    client = dask_client(config_path=parsed_args.config)
    dtrain = load_data(config_path=parsed_args.config)
    dtest = load_data(config_path=parsed_args.config, test=True)
    dtrain = feature_engg(dtrain)
    dtest = feature_engg(dtest)
    X_train, X_test, y_train, y_test = split_data(dtrain, config_path=parsed_args.config)
    dtrain, dvalid = generate_Dmatrix(client, X_train, X_test, y_train, y_test)
    _ = train_with_tracking(client, dtrain, dvalid, dtest, config_path=parsed_args.config)
    client.close()