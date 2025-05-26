import logging
import logging.config
import loguru
import inspect
from abc import ABC, abstractmethod
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',  
    datefmt='%Y-%m-%d %H:%M:%S'               
)

logger = logging.getLogger(__name__)

#finished
class BaseLogger(ABC):
    @abstractmethod
    def log(self, log_method: str, message: str):
        pass

    @abstractmethod
    def set_config(self, config: Dict):
        pass

    @staticmethod
    def _validate_log_method_params(self, log_method: str | None, message: str):
        if not isinstance(message, str):
            raise ValueError("Message must be a string.")
        if not isinstance(log_method, str):
            raise ValueError("Log level must be a string.")
    
    @staticmethod
    def _invoke_log_method(self, obj: object, log_method: str, message: str):
        log_method = log_method.lower()
        try:
            if hasattr(obj, log_method):
                getattr(obj, log_method)(message)
        except Exception:
            raise ValueError(f"Invalid value: {log_method}. Must be one of ['debug', 'info', 'warning', 'error', 'critical']")

    @staticmethod
    def _validate_set_config_method_params(self, config: Dict):
        if not isinstance(config, dict):
            raise ValueError("Config must be a dict")

#finished
class LoggingStrategy(BaseLogger):
    def __init__(self, logger_name: str, log_level: str = "DEBUG"):
        self.logger = logging.getLogger(logger_name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(getattr(logging, log_level.upper(), logging.DEBUG))

    def log(self, log_method: str, message: str):
        self._validate_log_method_params(log_method, message)
        self._invoke_log_method(self.logger, log_method, message)

    def set_config(self, config: Dict):
        self._validate_set_config_method_params(config)
        self._apply_config(config)

    @staticmethod
    def _apply_config(config):
        try:
            logging.config.dictConfig(config)
        except Exception:
            raise ValueError(f"Error when applying logging configuration for {config}.")

class LoguruStaregy(BaseLogger):
    def __init__(self):
        self.logger = loguru.logger

    def log(self, log_method: str, message: str):
        self._validate_log_method_params(log_method, message)
        self._invoke_log_method(self.logger, log_method, message)
        
class LoggingStrategyFactory:
    _strategies = {}
    
    @staticmethod
    def create():
        ...

    @staticmethod
    def get_strategy_mapping():
        ...

class LoggerProxy:
    def __init__(self, strategy: BaseLogger):
        self.levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        self._strategy = strategy
    def set_stategy(self, strategy: BaseLogger):
        self._strategy = strategy
        
    def __getattr__(self, level: str):
        if level.upper() in self.levels:
            def log_method(message: str):
                self._strategy.log(level.lower(), message)
            return log_method
                
        raise AttributeError(f"Level '{level}' not supported")


logger = LoggerProxy(LoggingStrategy(__name__))


# import inspect

# class BaseStrategy:
#     """Interface for all strategies."""
#     def execute(self):
#         raise NotImplementedError("Subclasses must implement `execute` method.")

# # Ví dụ các strategy
# class StrategyA(BaseStrategy):
#     def execute(self):
#         print("Executing Strategy A")

# class StrategyB(BaseStrategy):
#     def execute(self):
#         print("Executing Strategy B")

# class StrategyC(BaseStrategy):
#     def execute(self):
#         print("Executing Strategy C")

# class StrategyFactory:
#     """Factory for creating strategies based on class names."""
    
#     @classmethod
#     def get_strategy(cls, strategy_name: str) -> BaseStrategy:
#         # Lấy tất cả các class con của BaseStrategy
#         strategies = {
#             cls.__name__.lower(): cls for cls in BaseStrategy.__subclasses__()
#         }
        
#         # Tìm strategy theo tên, mặc định là StrategyA nếu không thấy
#         strategy_class = strategies.get(strategy_name.lower(), StrategyA)
#         return strategy_class()



# from abc import ABC, abstractmethod
# import logging
# from loguru import logger as loguru_logger
# from typing import Optional

# # Interface chung cho các chiến lược Logging
# class LoggerStrategy(ABC):
#     @abstractmethod
#     def log(self, level: str, message: str) -> None:
#         pass

# # Concrete Strategy: Sử dụng logging
# class LoggingStrategy(LoggerStrategy):
#     def __init__(self, logger_name: str = "default_logger", level: str = "DEBUG"):
#         self.logger = logging.getLogger(logger_name)
#         if not self.logger.handlers:  # Tránh thêm handler trùng lặp
#             handler = logging.StreamHandler()
#             formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
#             handler.setFormatter(formatter
#             self.logger.addHandler(handler)
#             self.logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))

#     def log(self, level: str, message: str) -> None:
#         if not isinstance(message, str)
#             raise ValueError("Message must be a string")
#         level = level.upper()
#         if hasattr(self.logger, level.lower()):
#             getattr(self.logger, level.lower())(message)
#         else:
#             self.logger.error(f"Unsupported log level: {level}")

# # Concrete Strategy: Sử dụng Loguru
# class LoguruStrategy(LoggerStrategy):
#     def __init__(self, level: str = "DEBUG"):
#         self.level = level.upper()

#     def log(self, level: str, message: str) -> None:
#         if not isinstance(message, str):
#             raise ValueError("Message must be a string")
#         level = level.upper()
#         if hasattr(loguru_logger, level.lower()):
#             getattr(loguru_logger, level.lower())(message)
#         else:
#             loguru_logger.error(f"Unsupported log level: {level}")

# # Context: Quản lý chiến lược logging
# class LoggerContext:
#     def __init__(self, strategy: LoggerStrategy):
#         self._strategy = strategy
#         self._levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

#     def set_strategy(self, strategy: LoggerStrategy) -> None:
#         self._strategy = strategy

#     def __getattr__(self, level: str):
#         if level.upper() in self._levels:
#             def log_method(message: str) -> None:
#                 self._strategy.log(level.upper(), message)
#             return log_method
#         raise AttributeError(f"Log level '{level}' not supported")

# # Sử dụng
# if __name__ == "__main__":
#     # Khởi tạo với logging
#     logger = LoggerContext(LoggingStrategy())
#     logger.info("Info from logging")
#     logger.error("Error from logging")

#     # Chuyển sang Loguru
#     logger.set_strategy(LoguruStrategy())
#     logger.info("Info from Loguru")
#     logger.error("Error from Loguru")
