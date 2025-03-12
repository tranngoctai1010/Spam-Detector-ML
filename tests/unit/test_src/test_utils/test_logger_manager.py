# Build-in imports
import os
import yaml
import logging
import logging.config
import sys

# Third-party imports
import pytest

# Internal imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from src.utils.logger_manager import LoggerManager

@pytest.fixture(autouse=True)
def reset_logger_manager():
    LoggerManager._is_setup = False
    logging.getLogger().handlers.clear()
    
def test_setup_logging_success(mocker):
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("yaml.safe_load", return_value={"version": 1, "handlers": {"console": {"class": "logging.StreamHandler"}}})
    LoggerManager.setup_logging()
    assert LoggerManager._is_setup

def test_setup_logging_file_not_found(mocker):
    mocker.patch("os.path.exists", return_value=False)
    with pytest.raises(FileNotFoundError):
        LoggerManager.setup_logging()

def test_get_logger(mocker):
    mocker.patch("LoggerManager.setup_logging")
    logger = LoggerManager.get_logger("test_logger")
    assert isinstance(logger, logging.Logger)