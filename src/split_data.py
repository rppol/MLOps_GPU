import argparse
from read_params import read_params
from dask_ml.model_selection import train_test_split

from dask_client import dask_client
from load_data import load_data
from feature_engg import feature_engg

def split_data(df, config_path):
    config = read_params(config_path)

    split_ratio = config["split_data"]["split_ratio"]
    random_state = config["base"]["random_state"]

    y = df[config["base"]["target_col"]]
    X = df.drop([config["base"]["target_col"]], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=random_state, shuffle=True)

    return (X_train, X_test, y_train, y_test)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    client = dask_client(config_path=parsed_args.config)
    df = load_data(config_path=parsed_args.config)
    df = feature_engg(df)
    _1, _2, _3, _4 = split_data(df, config_path=parsed_args.config)
    client.close()