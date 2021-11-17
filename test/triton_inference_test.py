import sys
sys.path.append('src/')
import argparse

from read_params import read_params

from dask_client import dask_client
from load_data import load_data
from feature_engg import feature_engg

import tritonclient.grpc as triton_grpc

"""sudo docker run   --gpus=all   --rm   -p 8000:8000   -p 8001:8001   -p 8002:8002  -v /var/lib/jenkins/workspace/model_repository:/models   triton_fil   tritonserver   --model-repository=/models --model-control-mode=poll --repository-poll-secs=10"""

def triton_inference(df, config_path):
    config = read_params(config_path)

    df = df.astype(config["triton"]["dtype"])
    df = df.sample(frac=0.01)
    df = df.drop([config["base"]["target_col"]], axis=1)
    df = df.compute().to_pandas().values

    grpc_client = triton_grpc.InferenceServerClient(
        url=config["triton"]["ip"] + ':' + config["triton"]["grpc_port"],
        verbose = False
    )

    # Set up Triton input and output objects for GRPC
    triton_input_grpc = triton_grpc.InferInput(
        'input__0',
        (df.shape[0], df.shape[1]),
        'FP32'
    )
    triton_input_grpc.set_data_from_numpy(df)
    triton_output_grpc = triton_grpc.InferRequestedOutput('output__0')

    request_grpc = grpc_client.infer(
        'fil',
        model_version='1',
        inputs=[triton_input_grpc],
        outputs=[triton_output_grpc]
    )

    # Get results as numpy arrays
    predictions = request_grpc.as_numpy('output__0')
    return predictions

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    client = dask_client(config_path=parsed_args.config)
    df = load_data(config_path=parsed_args.config, test = True)
    df = feature_engg(df)
    triton_inference(df, config_path=parsed_args.config)
    client.close()