from config import Settings, get_log_filename
from logging_utils import setup_logging
from ui import main

if __name__ == "__main__":
    log_file = get_log_filename()
    setup_logging(log_file)
    main(log_file)
