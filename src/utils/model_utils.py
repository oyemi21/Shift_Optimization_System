import logging
import os
import sys
from src.exception import MyException
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import dagshub
from src.logger import configure_logger
from config.constant import registered_model_name
from src.utils.mlflow_setup import setup_mlflow

setup_mlflow()
logging = configure_logger()

def get_existing_model_metrics(registered_model_name):
    try:
        client = MlflowClient()
        versions = client.get_latest_versions(registered_model_name)

        if not versions:
            return None, None
        
        best_r2, best_mae =None, None

        for v in versions:
            run_id = v.run_id
            run = client.get_run(run_id)
            r2 = run.data.metrics.get("r2_score")
            mae = run.data.metrics.get("mae_score")

            if r2 is None or mae is None:
                continue
            
            if (best_r2 is None) or (r2 > best_r2) or (r2 == best_r2 and mae < best_mae):
                best_r2 = r2
                best_mae = mae
        
        return best_r2, best_mae
    except Exception as e:
        logging.warning("No existing model or metrics was found")
        return None, None

def load_model_from_mlflow(model_name = registered_model_name):
    try:
        client = MlflowClient()

        logging.info(f"fetching the production model {model_name}")

        latest_versions = client.get_latest_versions(
            name = model_name,
            stages = ["Production"]
        )

        if not latest_versions:
            raise Exception("No production model found in the mlflow registry...")
        
        latest_version = latest_versions[0].version

        model_uri = f"models:/{model_name}/{latest_version}"

        model = mlflow.pyfunc.load_model(model_uri)

        logging.info("model successfully loaded from the production stage")
        return model
    
    except Exception as e:
        logging.error(f"error occurred while loading the model from mlflow {e}")
        raise MyException(e, sys)
