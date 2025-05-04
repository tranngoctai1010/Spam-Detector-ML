from abc import ABC, abstractmethod

class BaseTask(ABC):
    """Abstract base class for tasks."""
    
    @abstractmethod
    def get_models(self):
        """Return list of models for the task.
        
        Returns
        -------
        models : list
            List of model wrappers.
        """
        pass
    
    @abstractmethod
    def get_evaluator(self):
        """Return evaluator for the task.
        
        Returns  
        -------
        evaluator : object
            Evaluator instance.
        """
        pass













# Mô tả: Chứa class BaseTask, định nghĩa giao diện cho các tác vụ học máy (như phân loại, hồi quy).
# Vai trò: Đảm bảo mỗi tác vụ cung cấp danh sách mô hình và evaluator phù hợp.

# Giải thích:
# BaseTask yêu cầu hai phương thức:
# get_models(): Trả về danh sách wrapper mô hình cho tác vụ (như danh sách mô hình phân loại).
# get_evaluator(): Trả về evaluator phù hợp (như ClassificationEvaluator).
# Class này được kế thừa bởi các tác vụ trong tasks/ (như ClassificationTask, RegressionTask).
# Giao diện giúp AutoML biết tác vụ nào dùng mô hình và metric nào.