![GitHub](https://img.shields.io/github/license/rppol/MLOps_GPU)
# Accelerating MLOps with the power of GPUs

## Code Structure

 1.  [notebooks](https://github.com/rppol/MLOps_GPU/tree/main/notebooks)  : These notebooks will not be deployed, as they are for performance benchmarking & experimentation.
	 -   [nyc-taxi-fare-cpu.ipynb](https://github.com/rppol/MLOps_GPU/blob/main/notebooks/nyc-taxi-fare-cpu.ipynb) : An intuitive CPU only notebook built with pandas (Data Wrangling), modin (multithreading), XGBoost (all core training).
		-   [nyc-taxi-fare-gpu.ipynb](https://github.com/rppol/MLOps_GPU/blob/main/notebooks/nyc-taxi-fare-gpu.ipynb) : An intuitive GPU only notebook built with dask_cudf (Data Wrangling), xgboost.dask (distributed GPU based training).
 2. [src](https://github.com/rppol/MLOps_GPU/tree/main/src) : This folder contains all the source files. We have modularized above notebook to make it easy to configure for MLOps deployment. Every major function is split into intuitively named python scripts. Each script can be run independently.
	-   [read_params.py](https://github.com/rppol/MLOps_GPU/blob/main/src/read_params.py) : Changing hardcoded values can be very annoying & prone to errors. This python script builds a helper functions to read [params.yaml](https://github.com/rppol/MLOps_GPU/blob/main/params.yaml) file, i.e. a one stop destination to modify parameters.
	-   [dask_client.py](https://github.com/rppol/MLOps_GPU/blob/main/src/dask_client.py) : Builds the Dask Client, this client is carried through the scripts as creating a new one takes time. Maximum allocated GPU Memory must be specified in the [params.yaml](https://github.com/rppol/MLOps_GPU/blob/main/params.yaml) file.
	-   [load_data.py](https://github.com/rppol/MLOps_GPU/blob/main/src/load_data.py) : Helps in loading both Training & Test data in GPU Memory. Uses dask_cudf to speed up & distribute reading. “test” parameter must be set to true for reading test data. Relative path to the data directory & file names must be provided in the [params.yaml](https://github.com/rppol/MLOps_GPU/blob/main/params.yaml) file.
	-   [feature_engg.py](https://github.com/rppol/MLOps_GPU/blob/main/src/feature_engg.py) : This is a Data Cleaning, Manipulation & Feature Addition script. It removes outliers and unreasonable data and adds interesting features like distance to various landmarks.
	-   [split_data.py](https://github.com/rppol/MLOps_GPU/blob/main/src/split_data.py) : Splits the data as training & validation scripts. The split ratio can be varied using [params.yaml](https://github.com/rppol/MLOps_GPU/blob/main/params.yaml) file.
	-   [generate_Dmatrix.py](https://github.com/rppol/MLOps_GPU/blob/main/src/generate_Dmatrix.py) : Converts data to a format XGBoost can handle.
	-   [train.py](https://github.com/rppol/MLOps_GPU/blob/main/src/train.py) : Training script for local use & troubleshooting. This isn’t used in the automated pipeline.
	-   [train_with_tracking.py](https://github.com/rppol/MLOps_GPU/blob/main/src/train_with_tracking.py) : Model training script with MLflow tracking. MLflow server must be running before execution of this script. It tracks all the parameters & model built. It also tests the model on unseen data, and tracks metrics. Visit [this](https://towardsdatascience.com/xgboost-fine-tune-and-optimize-your-model-23d996fab663) for a guide on fine tuning XGBoost models.
	-   [log_production_model.py](https://github.com/rppol/MLOps_GPU/blob/main/src/log_production_model.py) : This script finds the best model out of all versions by comparing rmse. It loads, converts & saves this model in production model folder for further processing.
 3. [test](https://github.com/rppol/MLOps_GPU/tree/main/test)
	 -   [test_and_evaluate.py](https://github.com/rppol/MLOps_GPU/blob/main/test/test_and_evaluate.py) : Generates predictions on unseen data & compares with actual values. Generates metrics like Root Mean Squared Error, Mean Absolute Error, R-squared Error. Model can be passed as an argument or it’s loaded from production_model directory.
	-   [triton_inference_test.py](https://github.com/rppol/MLOps_GPU/blob/main/test/triton_inference_test.py) : Tests if Triton Server is working or not with 1% data. Uses grpc for networking.
 4. [scripts](https://github.com/rppol/MLOps_GPU/tree/main/scripts)
	-   [train_test_log.sh](https://github.com/rppol/MLOps_GPU/blob/main/scripts/train_test_log.sh) : Trains, Tests & Logs new model.
	-   [Productionize.sh](https://github.com/rppol/MLOps_GPU/blob/main/scripts/Productionize.sh) : Gets best Model & copies it to Model Repository.
	-   [triton_test.sh](https://github.com/rppol/MLOps_GPU/blob/main/scripts/triton_test.sh) : Tests Triton after new model is loaded.
 6. artifacts  : Stores models in MLflow format.
 7. [Jenkinsfile](https://github.com/rppol/MLOps_GPU/blob/main/Jenkinsfile) : Describes CI/CD Pipeline.
 8. [params.yaml](https://github.com/rppol/MLOps_GPU/blob/main/params.yaml) : One stop Destination to control everything, stores configuration, parameters, path, ip, ports.