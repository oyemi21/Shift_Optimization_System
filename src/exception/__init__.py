import os
import sys
import logging

def error_message_detail(error: Exception, error_detail: sys) -> str:
    # extracting the traceback details from the exception information
    _, _, exc_tb = error_detail.exc_info()
    # getting the filename where the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    # create the formatted error message with file name, line number, and error message

    # getting the line number where the error occurred
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in file: {file_name} at line: {line_number} with message: {str(error)}"

    logging.error(error_message)  # Log the error message
    return error_message

class MyException(Exception):
    def __init__(self, error_message: str, error_detail: sys):

        super().__init__(error_message)

        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        # return the error message in proper string format
        return self.error_message
    