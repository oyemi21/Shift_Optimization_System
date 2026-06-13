import logging
import os
from logging.handlers import RotatingFileHandler
from config.constant import log_file_path, MAX_LOG_SIZE, BACKUP_COUNT

def configure_logger():
    logger = logging.getLogger()
    
    # prevent duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
    )

    # Create a rotating file handler
    file_handler = RotatingFileHandler(
        log_file_path, 
        maxBytes=MAX_LOG_SIZE, 
        backupCount=BACKUP_COUNT
        )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console handler 
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

