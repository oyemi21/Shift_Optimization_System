import pandas as pd
import numpy as np
from src.logger import configure_logger
from src.exception import MyException

import sys

logging = configure_logger()

def prediction_pipeline(input_data: dict, model):
    """
    uses the preloaded model from mlflow, to return a prediction

    Input:
        Input_data: dictionary from the API
        model: preloaded model from mlflow
    Output:
        model prediction
    """
    try:
        logging.info("preparing input data for prediction...")
        df = pd.DataFrame([input_data])

        prediction = model.predict(df)

        logging.info("prediction complete")
        return prediction.tolist()
    
    except Exception as e:
        logging.error(f"Error occurred in prediction pipeline {e}")
        raise MyException(e, sys)