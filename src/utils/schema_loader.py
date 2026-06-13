import yaml
from src.logger import configure_logger
from src.exception import MyException
import sys

def read_yaml(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise MyException(e,sys)
    