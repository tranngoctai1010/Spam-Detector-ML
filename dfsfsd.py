import logging
import os
from typing import List, Optional
from abc import ABC, abstractmethod
from beartype import beartype

# Định nghĩa bộ lọc chung
class LevelFilter(logging.Filter):
    """Bộ lọc log dựa trên danh sách các mức độ log được chỉ định."""
    
    def __init__(self, levels: List[str]):
        super().__init__()
        self.levels = [level.upper() for level in levels]

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelname in self.levels

# Class quản lý logging
@beartype
class DualFileLoggerManager(ABC):
    _is_configured: bool = False

    @classmethod
    def setup_logging(cls, log_dir: str) -> None:
        """Thiết lập logging với hai file: info.log và error.log."""
        if cls._is_configured:
            return

        try:
            # Đường dẫn cho hai file log
            info_log_path = os.path.join(log_dir, "info.log")
            error_log_path = os.path.join(log_dir, "error.log")

            # Tạo thư mục nếu chưa tồn tại
            os.makedirs(log_dir, exist_ok=True)

            # Đảm bảo file tồn tại
            if not os.path.exists(info_log_path):
                open(info_log_path, "a").close()
            if not os.path.exists(error_log_path):
                open(error_log_path, "a").close()

            # Cấu hình logging
            config = {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "standard": {
                        "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                    },
                },
                "handlers": {
                    "info_handler": {
                        "class": "logging.FileHandler",
                        "level": "DEBUG",
                        "formatter": "standard",
                        "filename": info_log_path,
                        "filters": ["info_filter"]
                    },
                    "error_handler": {
                        "class": "logging.FileHandler",
                        "level": "WARNING",
                        "formatter": "standard",
                        "filename": error_log_path,
                        "filters": ["error_filter"]
                    },
                },
                "filters": {
                    "info_filter": {
                        "()": "__main__.LevelFilter",
                        "levels": ["DEBUG", "INFO"]
                    },
                    "error_filter": {
                        "()": "__main__.LevelFilter",
                        "levels": ["WARNING", "ERROR", "CRITICAL"]
                    },
                },
                "loggers": {
                    "": {  # Root logger
                        "level": "DEBUG",
                        "handlers": ["info_handler", "error_handler"],
                    },
                },
            }

            # Áp dụng cấu hình
            logging.config.dictConfig(config)
            cls._is_configured = True
        except Exception as e:
            cls._is_configured = False
            raise RuntimeError(f"Error setting up logging: {e}")

    @classmethod
    def get_logger(cls, log_dir: str, name: str | None = None) -> Optional[logging.Logger]:
        """Trả về một logger đã được cấu hình."""
        if not cls._is_configured:
            cls.setup_logging(log_dir)
        return logging.getLogger(name)

    @staticmethod
    @abstractmethod
    def _resolve_path(file_path: str, must_exist: bool) -> str:
        pass

# Triển khai cụ thể
@beartype
class SimpleDualFileLoggerManager(DualFileLoggerManager):
    @staticmethod
    def _resolve_path(file_path: str, must_exist: bool) -> str:
        absolute_path = os.path.abspath(file_path)
        if must_exist and not os.path.exists(absolute_path):
            open(absolute_path, "a").close()
        return absolute_path

# Cách sử dụng
if __name__ == "__main__":
    # Thiết lập logging
    logger_manager = SimpleDualFileLoggerManager()
    logger_manager.setup_logging(log_dir="./logs")
    logger = logger_manager.get_logger(log_dir="./logs", name="my_app")

    # Ghi log để kiểm tra
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")

    print("Logging setup complete. Check the files in ./logs/")