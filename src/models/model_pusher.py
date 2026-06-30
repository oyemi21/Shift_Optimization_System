from src.utils.model_utils import get_existing_model_metrics, load_model_from_mlflow
from src.utils.mlflow_setup import setup_mlflow
import sys
import os
from src.logger import configure_logger
from src.exception import MyException
from config.constant import registered_model_name
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import mlflow


logging = configure_logger()

class ModelPusher:
    def __init__(self):
        setup_mlflow()
        self.client = MlflowClient()
        self.registered_model_name = registered_model_name
    
    def push_model(self, model, r2_score: float, mae_score: float) -> bool:
        """
        Push model to mlflow registry and promote to production if it performs better
        """

        try:
            logging.info("Checking the existing model performanc...")

            old_r2, old_mae = get_existing_model_metrics(
                self.registered_model_name
            )

            push_new_model = False

            # Model Comparison logic
            if old_r2 is None:
                push_new_model = True
                logging.info("No existing model found, registering new model...")
            elif r2_score > old_r2:
                push_new_model = True
                logging.info(f"New model performed better than the old one")
            elif r2_score == old_r2 and mae_score < old_mae:
                push_new_model = True
                logging.info(f"Equal r2 scor but better mae score from the new model.")
            else:
                logging.info("Existing model is better than the new model. Skipping model registry and promotion")
                return False
            
            with mlflow.start_run():
                logging.info("logging and promoting the new model to mlflow...")
                mlflow.log_metric("r2_score", r2_score)
                mlflow.log_metric("mae_score", mae_score)

                mlflow.sklearn.log_model(
                    sk_model = model,
                    artifact_path = "model",
                    registered_model_name = self.registered_model_name
                )

                latest_versions = self.client.get_latest_versions(
                    name = self.registered_model_name,
                    stages = ["None"]
                )

                if not latest_versions:
                    raise Exception("No newly registered model found...")
        
                latest_version = latest_versions[0].version
                
                logging.info(f"promoting model version {latest_version} to production")

                #transitioning the registered model to production status
                self.client.transition_model_version_stage(
                    name = self.registered_model_name,
                    version= latest_version,
                    stage = "Production",
                    archive_existing_versions= True
                )

                logging.info(f"model version {latest_version} is now in production")

                return True
        
        except Exception as e:
            logging.error(f"error occurred while pushing and promoting the model {e}")
            raise MyException(e, sys)