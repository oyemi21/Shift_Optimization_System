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

logging = configure_logger()

class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train.squeeze()
        self.y_test = y_test.squeeze()
        self.schema = read_yaml(SCHEMA_PATH)
        self.model_pipeline = None

    def build_pipeline(self):
        """build pipeline and include the preprocesing method for categorical columns"""
        Numerical_columns = self.schema["column"]["Numerical_features"]
        Categorical_columns = self.schema["column"]["Categorical_features"]

        preprocess = ColumnTransformer(
            transformers = [
                ('cat', OneHotEncoder(handle_unknown ='ignore'), Categorical_columns),
                ('num', 'passthrough', Numerical_columns)
            ]
        )

        pipeline = Pipeline(steps=[
            ('pre_process', preprocess),
            ('model', RandomForestRegressor())
        ])

        return pipeline
    
    def train_model(self):
        try:
            logging.info("starting the model training pipeline...")
            self.model_pipeline = self.build_pipeline()
            self.model_pipeline.fit(self.X_train, self.y_train)
            logging.info("model training successful")
            return self.model_pipeline
        except Exception as e:
            logging.error(f"Error occurred during model training{e}")
            raise MyException(e, sys)
        
    def evaluate_model(self):
        try:
            logging.info("evaluating the model performance...")
            y_pred = self.model_pipeline.predict(self.X_test)
            r2 = r2_score(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            logging.info(f"pipeline evaluation done with the performance of {r2} r2score, {mae} mae score and {mse} mse score")
            return r2, mae
        except Exception as e:
            logging.error("error occurred while evaluating the model {e}")
            raise MyException( e, sys)
