import logging

def setup_logging(log_file: str):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='\n%(asctime)s | %(levelname)s | %(message)s\n'
    )

def log_section(tag: str, message: str):
    logging.info(f"\n{'='*20} [{tag}] {'='*20}\n{message}\n{'='*50}\n")
