import os
import argparse
from dask_client import read_params, dask_client
import cudf
import dask, dask_cudf

def load_train_data(config_path):
    config = read_params(config_path)
    client = dask_client(config_path)
    df = dask_cudf.read_csv(os.path.join("/home/nvidiatest/mlops_blog/data", "train.csv"))
    return (client, df)

def load_test_data(config_path):
    config = read_params(config_path)
    client = dask_client(config_path)
    df = dask_cudf.read_csv(os.path.join("/home/nvidiatest/mlops_blog/data", "test.csv"))
    return (client, df)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="/home/nvidiatest/mlops_blog/params.yaml")
    parsed_args = args.parse_args()
    load_train_data(config_path=parsed_args.config)