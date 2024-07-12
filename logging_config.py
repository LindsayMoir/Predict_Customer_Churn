import logging

def setup_logging(log_filename):
    """
    Set up logging configuration.

    Parameters:
    log_filename (str): The name of the log file.

    Returns:
    None
    """
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("Logging has been set up successfully")