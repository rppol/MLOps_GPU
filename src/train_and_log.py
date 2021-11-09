import os
import argparse
from dask_client import read_params
from load_data import load_train_data, load_test_data
from feature_engg import feature_engg
from split_data import split_data
from test_and_evaluate import eval_metrics, test_and_evaluate
from urllib.parse import urlparse
import xgboost as xgb
import numpy as np
import mlflow
import mlflow.xgboost

def train(client, dtrain, dvalid, config_path):
    config = read_params(config_path)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

    remote_server_uri = config["mlflow"]["remote_server_uri"]
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=config["mlflow"]["run_name"]):
        params = {
            'booster' : config["train"]["params"]["booster"],
            'eval_metric': config["train"]["params"]["eval_metric"],
            'tree_method':config["train"]["params"]["tree_method"],
            'objective': config["train"]["params"]["objective"],
            'min_child_weight': config["train"]["params"]["min_child_weight"],
            'colsample_bytree': config["train"]["params"]["colsample_bytree"],
            'learning_rate': config["train"]["params"]["learning_rate"],
            'max_depth': config["train"]["params"]["max_depth"]
        }

        num_boost_round = config["train"]["num_boost_round"]
        early_stopping_rounds = config["train"]["early_stopping_rounds"]
        verbose_eval = config["train"]["verbose_eval"]

        model = xgb.dask.train(client, params, dtrain,
                                num_boost_round=num_boost_round,
                                evals=watchlist,
                                early_stopping_rounds=early_stopping_rounds,
                                verbose_eval=verbose_eval)

        client, df = load_test_data(config_path)
        rmse, mae, r2 = test_and_evaluate(client, df, config_path)

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

    """mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 -p 8003"""

    if config["train"]["save_model"]:
        model_path = "saved_models"
        model['booster'].save_model(os.path.join(model_path, "xgboost.model"))
        model['booster'].save_model(os.path.join(model_path, "xgboost.json"))
    return model

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    client, df = load_train_data(config_path=parsed_args.config)
    client, df = feature_engg(client, df)
    client, dtrain, dvalid = split_data(client, df,config_path=parsed_args.config)
    train(client, dtrain, dvalid, config_path=parsed_args.config)