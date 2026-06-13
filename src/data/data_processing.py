import pandas as pd
from src.logger import configure_logger
from src.exception import MyException
from config.constant import target_column

import sys
from sklearn.model_selection import train_test_split
from src.data.data_ingestion import load_data
from src.data.data_validation import starting_datavalidation

logging  = configure_logger()

class DataProcessor:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        logging.info("data processor has been initialized...")

    def filling_missing_values(self):
        """
        this function fills the missing values in the data with the mean of the respective columns. 
        It logs an info message indicating that the missing values have been filled, and returns the data with filled missing values.
        """
        try:
            logging.info("filling missing values started...")
            self.data = self.data.sort_values(by='date')

            # fill temperature and humidity columns
            self.data['temperature'] = self.data['temperature'].fillna(
                method="ffill").fillna(self.data['temperature'].mean()
            )
            self.data['humidity'] = self.data['humidity'].fillna(
                method="ffill").fillna(self.data['humidity'].mean()
            )

            # Fill Timestamp
            self.data["timestamp"] = self.data["timestamp"].fillna(
                method="ffill"
            )

            # Fill the categorical fields

            self.data['issue_type'] = self.data['issue_type'].fillna('No Issue')
            self.data['resolved_by'] = self.data['resolved_by'].fillna('No Maintenance')

            # Fill the maintenance downtime with 0 for shifts without issues
            self.data['maintenance_downtime'] = self.data['maintenance_downtime'].fillna(0)

            self.data = self.data.drop(columns=['maintenance_id'])

            
            logging.info("filling missing values completed successfully...")
            
            return self.data
        except Exception as e:
            logging.error(f"Error occurred while filling missing values: {e}")
            raise MyException(e, sys)
        
    def removing_duplicates(self):
        try:
            duplicates = self.data.duplicated().sum()

            if duplicates > 0:
                logging.warning(f"Removing {duplicates} duplicated rows")
                self.data.drop_duplicates(inplace=True)

            return self.data

        except Exception as e:
            raise MyException(e, sys)
    
    def preprocess_data(self):
        try:
            logging.info("starting the data preprocessing pipeline")
            self.data = self.filling_missing_values()
            self.data = self.removing_duplicates()

            logging.info("data processing completed...")
            return self.data
        except Exception as e:
            logging.error("error occurred while processing the data...")
            raise MyException(e, sys)

    def split_X_y(self, data: pd.DataFrame):
        try:
            X = data.drop(columns=[target_column])
            y = data[target_column]
            logging.info(X.head())
            logging.info(y.head())
            logging.info("data has been successfully splitted into features and target columns")
            return X, y
        except Exception as e:
            logging.error("error occurred while splitting into features and target columns")
            raise MyException(e, sys)
        
    def train_test_splitting(self, data: pd.DataFrame):
        try:
            X, y = self.split_X_y(data)
            X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = 0.2, random_state = 42
            )
            logging.info("Train-test splitting completed.")

            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("error occurred while splitting the data into training and testing...")
            raise MyException(e, sys)
        
def start_data_preprocessing(data: pd.DataFrame):
    try:
        processor = DataProcessor(data)
        data = processor.preprocess_data()
        return data
    except Exception as e:
        raise MyException(e, sys)

# data = load_data()
# data = starting_datavalidation(data)
# start_data_preprocessing(data)
    