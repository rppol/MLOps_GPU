import os
import argparse
from dask_client import read_params
from load_data import load_train_data
from feature_engg import feature_engg
from split_data import split_data
import xgboost as xgb
import mlflow
import mlflow.xgboost

def train(client, dtrain, dvalid, config_path):
    config = read_params(config_path)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    with mlflow.start_run():
    params = {
        'booster' : config["train"]["params"]["booster"],
        'eval_metric': config["train"]["params"]["eval_metric"],
        'tree_method':config["train"]["params"]["tree_method"],
        'objective': config["train"]["params"]["objective"],
        'min_child_weight': config["train"]["params"]["min_child_weight"],
        'colsample_bytree': config["train"]["params"]["colsample_bytree"],
        'learning_rate': config["train"]["params"]["learning_rate"],
    }

    num_boost_round = config["train"]["num_boost_round"]
    early_stopping_rounds = config["train"]["early_stopping_rounds"]
    verbose_eval = config["train"]["verbose_eval"]

    model = xgb.dask.train(client, params, dtrain,
                            num_boost_round=num_boost_round,
                            evals=watchlist,
                            early_stopping_rounds=early_stopping_rounds,
                            verbose_eval=verbose_eval)

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