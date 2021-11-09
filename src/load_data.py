import os
import argparse
from read_params import read_params
import dask_cudf

from dask_client import dask_client

def load_data(config_path, test=False):
    config = read_params(config_path)

    data_dir = config["load_data"]["data_dir"]
    data = config["load_data"]["train_data_name"]
    if test:
        data = config["load_data"]["test_data_name"]

    df = dask_cudf.read_csv(os.path.join(data_dir, data))
    return df

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    client = dask_client(config_path=parsed_args.config)
    load_data(config_path=parsed_args.config)
    client.close()