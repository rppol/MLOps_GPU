from read_params import read_params
import argparse
import mlflow
from mlflow.tracking import MlflowClient
from pprint import pprint
import joblib
import os

def log_production_model(config_path):
    config = read_params(config_path)
    
    mlflow_config = config["mlflow"] 
    model_name = mlflow_config["registered_model_name"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    
    runs = mlflow.search_runs(experiment_ids=2, order_by=["metrics.rmse"])
    print(runs)
    lowest = runs["metrics.rmse"][0]
    print(lowest)
    lowest_run_id = runs[runs["metrics.rmse"] == lowest]["run_id"][0]
    
    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name}'"):
        mv = dict(mv)
        
        if mv["run_id"] == lowest_run_id:
            current_version = mv["version"]
            logged_model = mv["source"]
            pprint(mv, indent=4)
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Production"
            )
        else:
            current_version = mv["version"]
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Staging"
            )        

    loaded_model = mlflow.xgboost.load_model(logged_model)
    print("Loaded Model is : ", loaded_model)
    print("Type : ", type(loaded_model))
    
    model_path = "../saved_models"

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = log_production_model(config_path=parsed_args.config)