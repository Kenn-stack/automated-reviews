import logging
import os


log_dir = os.path.join(os.path.dirname(__file__), "logs")
def setup_logging(log_file="info.log", err_file="error.log"):
    
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Error handler
    error_handler = logging.FileHandler(os.path.join(log_dir, err_file))
    error_handler.setLevel(logging.ERROR)
    
    # Set formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    for h in [file_handler, console_handler, error_handler]:
        h.setFormatter(formatter)
    
    # Attach handlers
    logger.handlers = []
    for h in [file_handler, console_handler, error_handler]:
        logger.addHandler(h)
