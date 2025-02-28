# Build-in imports
import os
import yaml
import logging
import logging.config

# Third-party imports
import pytest

# Internal imports
from src.utils.logger_manager import LoggerManager


@pytest.fixture
def sample_logging_config():
    fixture_path = os.path.join(os.path.dirname(__file__), "..", "fixtures", "sample_logging_config.yaml")
    
    if not os.path.exists(fixture_path):
        config_content = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "standard",
                    "stream": "ext://sys.stdout"
                }
            },
            "loggers": {
                "": {
                    "level": "INFO",
                    "handlers": ["console"],
                    "propagate": True
                }
            }
        }
    with open(fixture_path, mode="r") as file:
        yaml.safe_load(config_content, file)
        
    return fixture_path


@pytest.fixture(autouse=True)
def reset_logger_manager():
    LoggerManager._is_setup = False
    logging.getLogger.handlers.clear()
    

def test_setup_logging_success(sample_logging_config, mocker):
    mocker.patch("os.path.dirname", return_value=os.path.dirname(sample_logging_config))
    mocker.patch("os.path.join", return_value=)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# import pytest
# import os
# import yaml
# import logging
# import logging.config
# from src.utils.logger_manager import LoggerManager

# # Fixture để load file logging_config.yaml từ fixtures/
# @pytest.fixture
# def sample_logging_config():
#     # Đường dẫn tới file mẫu trong fixtures/
#     fixture_path = os.path.join(os.path.dirname(__file__), "..", "fixtures", "sample_logging_config.yaml")
    
#     # Tạo file mẫu nếu chưa có (chỉ để test, trong thực tế bạn nên chuẩn bị file này trước)
#     if not os.path.exists(fixture_path):
#         config_content = {
#             "version": 1,
#             "disable_existing_loggers": False,
#             "formatters": {
#                 "standard": {
#                     "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#                 }
#             },
#             "handlers": {
#                 "console": {
#                     "class": "logging.StreamHandler",
#                     "level": "INFO",
#                     "formatter": "standard",
#                     "stream": "ext://sys.stdout"
#                 }
#             },
#             "loggers": {
#                 "": {
#                     "level": "INFO",
#                     "handlers": ["console"],
#                     "propagate": True
#                 }
#             }
#         }
#         with open(fixture_path, "w", encoding="utf-8") as f:
#             yaml.safe_dump(config_content, f)
    
#     return fixture_path

# # Reset trạng thái LoggerManager trước mỗi test
# @pytest.fixture(autouse=True)
# def reset_logger_manager():
#     LoggerManager._is_setup = False
#     logging.getLogger().handlers.clear()  # Xóa handlers để tránh xung đột

# # Test setup_logging thành công
# def test_setup_logging_success(sample_logging_config, mocker):
#     # Mock os.path để trả về file từ fixtures/
#     mocker.patch("os.path.dirname", return_value=os.path.dirname(sample_logging_config))
#     mocker.patch("os.path.join", return_value=sample_logging_config)

#     # Gọi setup_logging
#     LoggerManager.setup_logging()

#     # Kiểm tra trạng thái
#     assert LoggerManager._is_setup is True, "Logging should be set up"
    
#     # Kiểm tra logger hoạt động
#     logger = logging.getLogger("test")
#     assert len(logger.handlers) > 0, "Logger should have handlers after setup"

# # Test setup_logging khi file không tồn tại
# def test_setup_logging_file_not_found(mocker):
#     # Mock os.path để giả lập file không tồn tại
#     mocker.patch("os.path.exists", return_value=False)
#     mocker.patch("os.path.join", return_value="non_existent_path/logging_config.yaml")

#     with pytest.raises(FileNotFoundError) as exc_info:
#         LoggerManager.setup_logging()
    
#     assert str(exc_info.value) == "[LoggerManager][setup_logging] - The logging_config.yaml file was not found."
#     assert LoggerManager._is_setup is False, "Logging should not be set up"

# # Test setup_logging với file YAML lỗi cú pháp
# def test_setup_logging_invalid_yaml(mocker):
#     # Tạo file YAML lỗi trong fixtures/
#     fixture_path = os.path.join(os.path.dirname(__file__), "..", "fixtures", "invalid_logging_config.yaml")
#     with open(fixture_path, "w", encoding="utf-8") as f:
#         f.write("invalid: yaml: content: : :")  # YAML không hợp lệ

#     # Mock os.path
#     mocker.patch("os.path.dirname", return_value=os.path.dirname(fixture_path))
#     mocker.patch("os.path.join", return_value=fixture_path)

#     with pytest.raises(ValueError) as exc_info:
#         LoggerManager.setup_logging()
    
#     assert "[loggerManager][setup_logging] - Error when reading the logging_config.yaml file" in str(exc_info.value)
#     assert LoggerManager._is_setup is False, "Logging should not be set up"

# # Test get_logger tự động setup
# def test_get_logger_auto_setup(sample_logging_config, mocker):
#     # Mock os.path
#     mocker.patch("os.path.dirname", return_value=os.path.dirname(sample_logging_config))
#     mocker.patch("os.path.join", return_value=sample_logging_config)

#     # Gọi get_logger lần đầu
#     logger = LoggerManager.get_logger("test_logger")

#     # Kiểm tra
#     assert isinstance(logger, logging.Logger), "Should return a Logger instance"
#     assert logger.name == "test_logger", "Logger name should match"
#     assert LoggerManager._is_setup is True, "Logging should be set up automatically"

# # Test get_logger với name mặc định
# def test_get_logger_default_name(sample_logging_config, mocker):
#     # Mock os.path
#     mocker.patch("os.path.dirname", return_value=os.path.dirname(sample_logging_config))
#     mocker.patch("os.path.join", return_value=sample_logging_config)

#     # Gọi get_logger không truyền name
#     logger = LoggerManager.get_logger()

#     # Kiểm tra name mặc định
#     assert logger.name == "src.utils.logger_manager", "Default logger name should be module name"

# # Test get_logger sau khi đã setup
# def test_get_logger_after_setup(sample_logging_config, mocker):
#     # Mock os.path và setup trước
#     mocker.patch("os.path.dirname", return_value=os.path.dirname(sample_logging_config))
#     mocker.patch("os.path.join", return_value=sample_logging_config)
#     LoggerManager.setup_logging()

#     # Gọi get_logger
#     logger = LoggerManager.get_logger("after_setup")

#     # Kiểm tra
#     assert logger.name == "after_setup", "Logger name should match"
#     assert len(logger.handlers) > 0, "Logger should have handlers"