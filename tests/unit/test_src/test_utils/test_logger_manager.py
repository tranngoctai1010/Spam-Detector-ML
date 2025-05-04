# Build in imports
import os
import yaml
import logging
import logging.config

# Third-party imports
import pytest

# Internal imports
from src.utils.logger_manager import LoggerManager

# pytest tests/unit/test_src/test_utils/test_logger_manager.py

@pytest.fixture(autouse=True)
def reset_logger_manager():
    LoggerManager.is_setup = False
    logging.getLogger().handlers.clear()

def test_setup_logging_success(mocker):
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("yaml.safe_load", return_value={"version": 1, "handlers": {"console": {"class": "logging.StreamHandler"}}})
    LoggerManager.setup_logging()
    assert LoggerManager._is_setup

# def test_setup_logging_file_not_found(mocker):
#     mocker.patch("os.path.exists", return_value=False)
#     with pytest.raises(FileNotFoundError): 
#         LoggerManager.setup_logging()

def test_get_logger(mocker):
    mocker.patch("src.utils.logger_manager.LoggerManager.setup_logging")
    logger = LoggerManager.get_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"