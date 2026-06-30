import dagshub
import mlflow
from config.constant import EXPERIMENT_NAME

def setup_mlflow():
    dagshub.init(
        repo_owner='oyemi21',
        repo_name='Shift_Optimization_System', 
        mlflow=True
    )
    mlflow.set_experiment(EXPERIMENT_NAME)