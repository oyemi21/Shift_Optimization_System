import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import sys

from config.constant import database_path
from src.logger import configure_logger
from src.exception import MyException   

logging  = configure_logger()

def load_data():
    """"
        Load data from a database and return it as a pandas DataFrame for other pipeline to be able to make use of it.
    """
    try:
        logging.info("data ingestion initiated...")
        logging.info("loading data from database file...")
        # initiating the connection to the database file.
        connection = sqlite3.connect(database_path)
        #loading data into a pandas DataFrame
        Shift_Data = pd.read_sql("SELECT * FROM ShiftPerformance", connection)

        logging.info(Shift_Data.head())
        logging.info("data loaded successfully...")

        return Shift_Data
    except Exception as e:
        logging.error(f"Error occurred while loading data: {e}")
        raise MyException(e, sys)

# load_data()