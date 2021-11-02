import os
import argparse
from dask_client import read_params, dask_client
import cudf
import dask, dask_cudf

def load_data(config_path):
    config = read_params(config_path)
    client = dask_client(config_path)
    df = dask_cudf.read_csv(os.path.join("data", "train.csv"))
    return (client, df)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_data(config_path=parsed_args.config)