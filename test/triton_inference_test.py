import os
import yaml
import argparse
import numpy as np
import dask, dask_cudf
from dask_cuda import LocalCUDACluster
from dask.delayed import delayed
from dask.distributed import Client, wait, LocalCluster
from dask.utils import parse_bytes
import tritonclient.http as triton_http
import tritonclient.grpc as triton_grpc
import math
from math import cos, sin, asin, sqrt, pi

"""sudo docker run   --gpus=all   --rm   -p 8000:8000   -p 8001:8001   -p 8002:8002   -v /home/nvidiatest/mlops_blog/model_repository:/models   triton_fil   tritonserver   --model-repository=/models"""

def jfk_distance(dropoff_latitude, dropoff_longitude, jfk_distance):
    for i, (x_1, y_1) in enumerate(zip(dropoff_latitude, dropoff_longitude)):
        x_1 = pi/180 * x_1
        y_1 = pi/180 * y_1
        x_jfk = pi/180 * 40.6413
        y_jfk = pi/180 * -73.7781
        
        dlon = y_jfk - y_1
        dlat = x_jfk - x_1
        a = sin(dlat/2)**2 + cos(x_1) * cos(x_jfk) * sin(dlon/2)**2
        
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers
        
        jfk_distance[i] = c * r
        
def lga_distance(dropoff_latitude, dropoff_longitude, lga_distance):
    for i, (x_1, y_1) in enumerate(zip(dropoff_latitude, dropoff_longitude)):
        x_1 = pi/180 * x_1
        y_1 = pi/180 * y_1
        x_lga = pi/180 * 40.7769
        y_lga = pi/180 * -73.8740
        
        dlon = y_lga - y_1
        dlat = x_lga - x_1
        a = sin(dlat/2)**2 + cos(x_1) * cos(x_lga) * sin(dlon/2)**2
        
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers
        
        lga_distance[i] = c * r
        
def ewr_distance(dropoff_latitude, dropoff_longitude, ewr_distance):
    for i, (x_1, y_1) in enumerate(zip(dropoff_latitude, dropoff_longitude)):
        x_1 = pi/180 * x_1
        y_1 = pi/180 * y_1
        x_ewr = pi/180 * 40.6895
        y_ewr = pi/180 * -74.1745
        
        dlon = y_ewr - y_1
        dlat = x_ewr - x_1
        a = sin(dlat/2)**2 + cos(x_1) * cos(x_ewr) * sin(dlon/2)**2
        
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers
        
        ewr_distance[i] = c * r
        
def tsq_distance(dropoff_latitude, dropoff_longitude, tsq_distance):
    for i, (x_1, y_1) in enumerate(zip(dropoff_latitude, dropoff_longitude)):
        x_1 = pi/180 * x_1
        y_1 = pi/180 * y_1
        x_tsq = pi/180 * 40.7580
        y_tsq = pi/180 * -73.9855
        
        dlon = y_tsq - y_1
        dlat = x_tsq - x_1
        a = sin(dlat/2)**2 + cos(x_1) * cos(x_tsq) * sin(dlon/2)**2
        
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers
        
        tsq_distance[i] = c * r
        
def met_distance(dropoff_latitude, dropoff_longitude, met_distance):
    for i, (x_1, y_1) in enumerate(zip(dropoff_latitude, dropoff_longitude)):
        x_1 = pi/180 * x_1
        y_1 = pi/180 * y_1
        x_met = pi/180 * 40.7794
        y_met = pi/180 * -73.9632
        
        dlon = y_met - y_1
        dlat = x_met - x_1
        a = sin(dlat/2)**2 + cos(x_1) * cos(x_met) * sin(dlon/2)**2
        
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers
        
        met_distance[i] = c * r
        
def wtc_distance(dropoff_latitude, dropoff_longitude, wtc_distance):
    for i, (x_1, y_1) in enumerate(zip(dropoff_latitude, dropoff_longitude)):
        x_1 = pi/180 * x_1
        y_1 = pi/180 * y_1
        x_wtc = pi/180 * 40.7126
        y_wtc = pi/180 * -74.0099
        
        dlon = y_wtc - y_1
        dlat = x_wtc - x_1
        a = sin(dlat/2)**2 + cos(x_1) * cos(x_wtc) * sin(dlon/2)**2
        
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers
        
        wtc_distance[i] = c * r

def add_features(df):
    df['hour'] = df['pickup_datetime'].dt.hour
    df['year'] = df['pickup_datetime'].dt.year
    df['month'] = df['pickup_datetime'].dt.month
    df['day'] = df['pickup_datetime'].dt.day
    df['weekday'] = df['pickup_datetime'].dt.weekday
    
    df = df.apply_rows(jfk_distance, incols=['dropoff_latitude', 'dropoff_longitude'],
                       outcols=dict(jfk_distance=np.float32), kwargs=dict())
    
    df = df.apply_rows(lga_distance, incols=['dropoff_latitude', 'dropoff_longitude'],
                       outcols=dict(lga_distance=np.float32), kwargs=dict())
        
    df = df.apply_rows(ewr_distance, incols=['dropoff_latitude', 'dropoff_longitude'],
                       outcols=dict(ewr_distance=np.float32), kwargs=dict())
            
    df = df.apply_rows(tsq_distance, incols=['dropoff_latitude', 'dropoff_longitude'],
                       outcols=dict(tsq_distance=np.float32), kwargs=dict())
    
    df = df.apply_rows(met_distance, incols=['dropoff_latitude', 'dropoff_longitude'],
                       outcols=dict(met_distance=np.float32), kwargs=dict())
    
    df = df.apply_rows(wtc_distance, incols=['dropoff_latitude', 'dropoff_longitude'],
                       outcols=dict(wtc_distance=np.float32), kwargs=dict())
    
    df = df.drop(['pickup_datetime','key'], axis=1)
    
    return df

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def triton_inference(config_path):
    config = read_params(config_path)

    http_client = triton_http.InferenceServerClient(
        url=config["triton"]["ip"] + ':' + config["triton"]["http_port"],
        verbose=False,
        concurrency=12
    )
    grpc_client = triton_grpc.InferenceServerClient(
        url=config["triton"]["ip"] + ':' + config["triton"]["grpc_port"],
        verbose = False
    )

    test = dask_cudf.read_csv('/home/nvidiatest/mlops_blog/data/test.csv')
    test = test.drop(['fare_amount'], axis=1)

    test['key'] = test['key'].astype('datetime64[ns]')
    test['pickup_datetime'] = test['pickup_datetime'].astype('datetime64[ns]')
    test['pickup_longitude'] = test ['pickup_longitude'].astype('float32')
    test['pickup_latitude'] = test ['pickup_latitude'].astype('float32')
    test['dropoff_longitude'] = test ['dropoff_longitude'].astype('float32')
    test['dropoff_latitude'] = test ['dropoff_latitude'].astype('float32')
    test['passenger_count'] = test ['passenger_count'].astype('uint8')

    # now add the features
    tparts = [dask.delayed(add_features)(part) for part in test.to_delayed()]
    test = dask_cudf.from_delayed(tparts)
    test = test.astype(config["triton"]["dtype"])
    test = test.compute().to_pandas().values

    # Set up Triton input and output objects for both HTTP and GRPC
    triton_input_http = triton_http.InferInput(
        'input__0',
        (test.shape[0], test.shape[1]),
        'FP32'
    )
    triton_input_http.set_data_from_numpy(test)
    triton_output_http = triton_http.InferRequestedOutput(
        'output__0',
    )
    triton_input_grpc = triton_grpc.InferInput(
        'input__0',
        (test.shape[0], test.shape[1]),
        'FP32'
    )
    triton_input_grpc.set_data_from_numpy(test)
    triton_output_grpc = triton_grpc.InferRequestedOutput('output__0')

    request_http = http_client.infer(
    'fil',
    model_version='1',
    inputs=[triton_input_http],
    outputs=[triton_output_http]
    )

    request_grpc = grpc_client.infer(
        'fil',
        model_version='1',
        inputs=[triton_input_grpc],
        outputs=[triton_output_grpc]
    )

    # Get results as numpy arrays
    result_http = request_http.as_numpy('output__0')
    result_grpc = request_grpc.as_numpy('output__0')

    # Check that we got the same result with both GRPC and HTTP
    np.testing.assert_almost_equal(result_http, result_grpc)

    print(result_grpc)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    triton_inference(config_path=parsed_args.config)