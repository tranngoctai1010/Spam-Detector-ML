import traceback
import functools
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import TypeVar, Self, Optional, Any
import logging

from beartype import beartype

from auto_ml.utils.logger_manager import BaseLoggerManager


TLoggerManager = TypeVar("TLoggerManager", bound="BaseLoggerManager")

@beartype
class BaseExceptionHandler(AbstractContextManager, ABC):
    """
    ---
    **Abstract class for exception handler.**

    _This class provides essential methods for managing and logging exceptions within an application.
    It supports logger configuration and utilizes a cache to optimize performance._

    ---
    ### Class attributes:
    
    - _default_logger_manager (type[TLoggerManager]): Default logger manager class.
    - _default_config_path (str): Default path for logger configuration.
    - _default_log_path (str): Path to the default log file.
    - _logger_cache (dict): A dictionary that caches loggers using log paths as keys.
    - _name (str | None = None): Logger name. Defaults to the root logger if not provided.

    ---
    ### Methods:

    - __init__(log_prefix, raise_exception, new_logger_manager, new_config_path, new_log_path):  
        _Initializes an instance with customizable configuration parameters._

    - __enter__():  
      _Abstract method for handling resource setup when entering a context manager._

    - __exit__():  
      _Abstract method for handling cleanup operations when exiting a context manager._

    - log_exception(func) (classmethod):  
      Logs exceptions occurring in a decorated function.

    - set_default_configuration(logger_manager, config_path, log_path) (classmethod):   
      Sets the default configuration for the logger.

    - _get_logger(new_logger_manager=None, new_config_path=None, new_log_path=None) (classmethod):  
      Returns a configured logger, using cache to improve performance.
    """
    _default_logger_manager: type[TLoggerManager]
    _default_config_path: str
    _default_log_path: str
    _logger_cache: dict = {}
    _name: str | None = None

    def __init__(
        self,
        log_prefix: str,
        raise_exception: bool = True,
        new_logger_manager: type[TLoggerManager] | None = None,
        new_config_path: str | None = None,
        new_log_path: str | None = None
    ):
        """
        **Initialize the instance for exception handling**

        ### Args: 

        - log_prefix (str): Prefix for log messages.
        - raise_exception (bool, optional): Dertermines whether to raise an exception after logging. Default to True.
        - new_logger_manager (type[TLoggerManager] | None, optional): Custom a new logger manager class. Default to None.
        - new_config_path (str | None, optional): Path to a new custom logger configuration file. Default to None.
        - new_log_path (str | None, optional): Path to a new custom log file. Default to None.
        """
        self.log_prefix = log_prefix
        self.raise_exception = raise_exception
        self.logging_logger = self._get_logger(new_logger_manager, new_config_path, new_log_path)

    @abstractmethod
    def __enter__(self) -> Self:
        """
        **Abstract method for handling resource setup when entering a context manager.**

        ### Returns:

        - self: The instance of the class.
        """
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        **Abstract method for handling cleanup operations when exiting a context manager.**

        ### Args:

        - exc_type (type[Exception] | None): The type of the exception, if any.
        - exc_value (Exception | None): The exception object, if any.
        - exc_traceback (traceback | None): The traceback object, if any.
        """
        pass

    @classmethod
    @abstractmethod
    def log_exception(func):
        """
        **Abstract class method to log exceptions occurring in a decorated function.**
        
        ### Args:

        - func (Callable):The function which exceptions should be logged.
        """
        pass

    @classmethod
    def set_default_configuration(
        cls,
        logger_manager: type[TLoggerManager],
        config_path: str,
        log_path: str,
        name: str | None = None
    ) -> None:
        """
        **Sets the default configuration for the logger.**

        ### Args:

        - logger_manager (type[TLoggerManager]): The logger management class to use.
        - config_path (str): The path to the configuration file for the logger.
        - log_path (str): The path to the log file.
        - name (str | None = None): Logger name. Defaults to the root logger if not provided
        """
        cls._default_logger_manager = logger_manager
        cls._default_config_path = config_path
        cls._default_log_path = log_path
        cls._name = name

    @classmethod
    def _get_logger(
        cls,
        new_logger_manager: type[TLoggerManager] | None = None,
        new_config_path: str | None = None,
        new_log_path: str | None = None
    ) -> Optional[logging.Logger]:
        """
        **Returns a configured logger, using cache to improve performance.**
   
        ### Args:

        - new_logger_manager (type[TLoggerManager] | None, optional): Custom logger manager class. Defaults to None.
        - new_config_path (str | None, optional): Custom configuration file path. Defaults to None.
        - new_log_path (str | None, optional): Custom log file path. Defaults to None.
        """
        target_logger_manager = new_logger_manager or cls._default_logger_manager
        target_config_path = new_config_path or cls._default_config_path
        target_log_path = new_log_path or cls._default_log_path
        if target_log_path not in cls._logger_cache:
            cls._logger_cache[target_log_path] = target_logger_manager.get_logger(target_config_path, target_log_path, cls._name)
        return cls._logger_cache[target_log_path]


@beartype
class PythonExceptionHandler(BaseExceptionHandler):
    """
    ---
    **Advanced exception handler in Python.**  

    _This class helps log errors in detail and control exception handling behavior._  
    It supports:  
        - Log errors in a systematic and organized manner.  
        - Control whether to continue raising exceptions (`raise_exception`)  
        - Use as a context manager (`with` statement)  
        - Decorator to automatically log errors in functions.

    ---
    ### Example:

    ```python
    >>> from src.utils.logger_manager import PythonLoggerManager
    >>> from src.utils.exception_handler import PythonExceptionHandler

    >>> # Set default configuration for exception handler
    >>> PythonExceptionHandler.set_default_configuration(
    ... PythonLoggerManager, "config.yaml", "src.log"
    ... )

    # Decorator to log exceptions in the function
    >>> @PythonExceptionHandler.log_exception
    >>> def add_numbers(a, b):
    ... return a + b

    >>> # Triggering the exception
    >>> sum = add_numbers(5, "6")
    ```
    """
    def __init__(
        self,
        log_prefix: str,
        raise_exception: bool = True,
        new_logger_manager: type[TLoggerManager] | None = None,
        new_config_path: str | None = None,
        new_log_path: str | None = None
    ):
        """
        **Initialize the instance for exception handling**

        ### Args:

        - log_prefix (str): Prefix for log messages.  
        - raise_exception (bool, optional): Dertermines whether to raise an exception after logging. Default to True.
        - new_logger_manager (type[TLoggerManager] | None, optional): Custom a new logger manager class. Default to None.
        - new_config_path (str | None, optional): Path to a new custom logger configuration file. Default to None.
        - new_log_path (str | None, optional): Path to a new custom log file. Default to None.
        """
        super().__init__(
            log_prefix,
            raise_exception,
            new_logger_manager,
            new_config_path,
            new_log_path
        )

    def __enter__(self):
        """
        **Allow the class to be used as a context manager.**

        ### Returns:

        - PythonExceptionHandler: Returns the object itself to be used in the with block.
        """
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        **Handle exceptions when exiting the with block. If an error occurs, the error information will be logged.
        The next behavior depends on raise_exception:**
        - If raise_exception = False, the error will be suppressed.
        - If raise_exception = True, the error will continue to be raised.

        ### Args:

        - exc_type (Type[Exception] | None): The exception type, if any.
        - exc_value (Exception | None): The exception object, if any.
        - exc_traceback (traceback | None): The traceback information of the error, if any.

        ### Returns:
            
        - bool: Returns True if the error should be suppressed, False if the error should continue to be raised.
        """
        if exc_type is not None:
            self.logging_logger.error(
                "%s, Error %s: %s\n%s",
                self.log_prefix,
                exc_type.__name__,
                exc_type,
                traceback.format_exc()
            )
        return not self.raise_exception

    @classmethod
    def log_exception(cls, func):
        """
        **Decorator to automatically log exceptions in functions.**

        _**If the function wrapped by this decorator encounters an error, the error information along with the full stack trace will be logged.
        After logging, the error will still be raised to avoid silent failure.**_

        ### Args:

        - func (Callable): The function to be wrapped for exception tracking.

        ### Returns:
            
        - Callable: The wrapped function that automatically logs errors.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = cls._get_logger(cls._default_logger_manager, cls._default_config_path, cls._default_log_path)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Xác định tiền tố log dựa trên loại hàm (class method hay standalone)
                if args and hasattr(args[0], '__class__'):
                    class_name = args[0].__class__.__name__
                    log_prefix = f"[{class_name}.{func.__name__}]"
                else:
                    log_prefix = f"[{func.__name__}]"
                
                logger.error("%s - Error %s: %s\n%s", log_prefix, type(e).__name__, e, traceback.format_exc())
                raise  # Luôn ném lại lỗi để tránh silent failures
        return wrapper
           






# # Decorator log lỗi đơn giản
# def log_error(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         try:
#             return func(*args, **kwargs)
#         except Exception as e:
#             # Lấy tên hàm
#             func_name = func.__name__
#             # Kiểm tra xem có class không
#             prefix = f"[{func_name}]"
#             if args and hasattr(args[0], '__class__'):
#                 class_name = args[0].__class__.__name__
#                 prefix = f"[{class_name}.{func_name}]"
#             # Log lỗi
#             logger.error(f"{prefix} Lỗi: {str(e)}")
#             raise  # Ném lại lỗi nếu cần
#     return wrapper