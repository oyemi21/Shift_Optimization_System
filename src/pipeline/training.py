from src.utils.schema_loader import read_yaml
from config.constant import SCHEMA_PATH 
import sys
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.logger import configure_logger
from src.data.data_ingestion import load_data
from src.data.data_validation import starting_datavalidation
from src.data.data_processing import start_data_preprocessing
from src.features.feature_engineering import start_feature_engineering
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.exception import MyException
from src.models.model_pusher import ModelPusher
from src.models.model_training import ModelTrainer

logging = configure_logger()

def start_model_training():
    try:
        raw_data = load_data()
        validated_data = starting_datavalidation(raw_data)
        processed_data = start_data_preprocessing(validated_data)
        X_train, X_test, y_train, y_test = start_feature_engineering(processed_data)
        trainer = ModelTrainer(X_train, X_test, y_train, y_test)
        pipeline = trainer.train_model()
        r2, mae = trainer.evaluate_model()

        model_pusher = ModelPusher()
        Model_was_registered = model_pusher.push_model(
            model = pipeline,
            r2_score = r2,
            mae_score = mae
        )

        if Model_was_registered:
            logging.info("New Pipeline registered and promoted in Mlflow...")
        else:
            logging.info("Existing model performs better than the new model..")
        return r2, mae, pipeline


    except Exception as e:
        logging.error(f"error occurred while starting the model training.")
        raise MyException(e,sys)

#start_model_training()