import pandas as pd
import numpy as np
import os
from src.logger import configure_logger
from src.exception import MyException
from config.constant import target_column
import sys
from src.utils.schema_loader import read_yaml
from src.data.data_ingestion import load_data
from src.data.data_validation import starting_datavalidation
from src.data.data_processing import DataProcessor
from config.constant import feature_artifact, SCHEMA_PATH
from src.data.data_processing import start_data_preprocessing

logging = configure_logger()

class FeatureEngineer:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        logging.info("feature engineering initialized...")
    
    def engineer_features(self):
        try:
            logging.info("started feature engineering process...")

            # Engineering new features
            self.data['start_time'] = pd.to_datetime(self.data['start_time'])
            self.data['end_time'] = pd.to_datetime(self.data['end_time'])
            self.data['date'] = pd.to_datetime(self.data['date'])

            # fixing the overnight shift and add 1 day where the end time < start time
            mask = self.data['end_time'] < self.data['start_time']
            self.data.loc[mask, 'end_time'] = self.data.loc[mask, 'end_time'] + pd.Timedelta(days=1)

            # Shift duration
            self.data['shift_duration'] = (self.data['end_time'] - self.data['start_time']).dt.total_seconds() / 3600

            # Defect rate
            self.data['defect_rate'] = self.data['defect_count'] / self.data['units_produced'].replace(0, pd.NA)

            self.data['downtime_ratio'] = self.data['downtime_minutes'] / (self.data['shift_duration'] * 60)

            # temporal feature
            self.data['day_of_week'] = self.data['date'].dt.dayofweek
            self.data['hour_of_day'] = self.data['start_time'].dt.hour

            # cyclical encoding
            self.data['day_of_week_sin'] = np.sin(2 * np.pi * self.data['day_of_week'] / 7)
            self.data['day_of_week_cos'] = np.cos(2 * np.pi * self.data['day_of_week'] / 7)
            self.data['hour_of_day_sin'] = np.sin(2 * np.pi * self.data['hour_of_day'] / 24)
            self.data['hour_of_day_cos'] = np.cos(2 * np.pi * self.data['hour_of_day'] / 24)

            logging.info("features successfully created...")
            return self.data
        except Exception as e:
            raise MyException(e, sys)
    
    def drop_unnecessary_columns_and_duplicates(self):
        try:
            schema = read_yaml(SCHEMA_PATH)
            columns_to_drop = schema["column"]["columns_to_drop"]

            existing_cols = [column for column in columns_to_drop if column in self.data.columns]
            self.data.drop(columns = existing_cols, inplace = True)

            processor = DataProcessor(self.data)
            self.data = processor.removing_duplicates()
            logging.info("dropped unnecessary columns and removed duplicates...")

            return self.data
        except Exception as e:
            raise MyException(e, sys)
    
    def feature_engine(self):
        try:
            self.data = self.engineer_features()
            self.data = self.drop_unnecessary_columns_and_duplicates()

            return self.data
        except Exception as e:
            raise MyException(e, sys)
        
def start_feature_engineering(data: pd.DataFrame):
    try:
        engineer = FeatureEngineer(data)
        data = engineer.feature_engine()

        # call the function that splits data from data processor
        processor = DataProcessor(data)
        X_train, X_test, y_train, y_test = processor.train_test_splitting(data)
        os.makedirs(os.path.dirname(feature_artifact), exist_ok = True)
        data.to_csv(feature_artifact, index= False)

        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise MyException(e, sys)

raw_data = load_data()
validated_data = starting_datavalidation(raw_data)
processed_data = start_data_preprocessing(validated_data)
    
X_train, X_test, y_train, y_test = start_feature_engineering(processed_data)
print(X_train.head())
print(X_train.shape)
print(y_train.head())