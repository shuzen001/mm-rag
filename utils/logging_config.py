import logging

DEFAULT_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(level: int = DEFAULT_LEVEL) -> None:
    """Configure basic logging for the application."""
    logging.basicConfig(level=level, format=LOG_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the given name."""
    return logging.getLogger(name)


# Configure logging on module import
setup_logging()
