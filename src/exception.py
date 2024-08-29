import sys
from src.logger import logging

def capture_error_details(error, error_detail):
    _, _, exc_tb = error_detail
    filename = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in file: {filename}, at line: {exc_tb.tb_lineno}, with message: {str(error)}"
    
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = capture_error_details(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message



if __name__ == '__main__':
    try:
        a=1/0
    except Exception as e:
        logging.info('Division by zero error')
        raise CustomException(e, sys.exc_info()) 
