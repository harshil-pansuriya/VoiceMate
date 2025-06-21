import logging
import os

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Ensure logs directory exists
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    # Set up file handler
    log_file = os.path.join(log_dir, 'app.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Add handler only if not already present
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
    
    return logger

# Initiate the logger instance
logger = setup_logger()