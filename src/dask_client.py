import argparse
import dask
from read_params import read_params
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from dask.utils import parse_bytes

def dask_client(config_path):
    config = read_params(config_path)
    max_memory = config["GPU_cluster"]["max_memory"]
    cluster = LocalCUDACluster(rmm_pool_size=parse_bytes(max_memory))   
    client = Client(cluster)
    dask.config.set({'distributed.scheduler.work-stealing': False})
    client.restart()
    return client

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    client = dask_client(config_path=parsed_args.config)
    client.close()
    