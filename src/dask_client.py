import os
import yaml
import argparse
import dask
from dask_cuda import LocalCUDACluster
from dask.distributed import Client, wait, LocalCluster
from dask.utils import parse_bytes

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def dask_client(config_path):
    config = read_params(config_path)
    max_memory = config["GPU_cluster"]["max_memory"]
    cluster = LocalCUDACluster(rmm_pool_size=parse_bytes(max_memory))   
    client = Client(cluster)
    dask.config.set({'distributed.scheduler.work-stealing': False})
    client.restart()
    #print(client)
    return client

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    client = dask
    dask_client(config_path=parsed_args.config)