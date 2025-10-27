import sys
from src.logger import logging

def error_message_detail(error: Exception):
    """
    Create detailed error message with filename and line number.
    """
    exc_type, exc_value, exc_tb = sys.exc_info()
    if exc_tb:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "Unknown"
        line_number = "Unknown"

    return f"Error in [{file_name}] line [{line_number}]: {str(error)}"


class CustomException(Exception):
    def __init__(self, error_message: Exception):
        detailed = error_message_detail(error_message)
        logging.error(detailed)
        super().__init__(detailed)
        self.error_message = detailed

    def __str__(self):
        return self.error_message
