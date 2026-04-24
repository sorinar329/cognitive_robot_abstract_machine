__version__ = "1.0.32"
import logging

from enum import Enum

logger = logging.Logger("segmind")
logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

# Handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)  # <-- this filters out DEBUG messages
logger.addHandler(handler)


class LogLevel(Enum):
    """
    Just to give a type to logging levels for type hints.
    """

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


def set_logger_level(level: LogLevel):
    """
    Because logging levels are filtered by the handler, we need to set the level of the handler. While
    we keep the logger level at DEBUG to allow all messages to pass through.
    """
    handler.setLevel(level.value)