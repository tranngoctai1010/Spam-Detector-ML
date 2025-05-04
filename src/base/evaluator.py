from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    """Abstract base class for evaluators."""
    
    @abstractmethod
    def evaluate(self, y_true, y_pred):
        """Evaluate predictions.
        
        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True labels or values.
        y_pred : array-like of shape (n_samples,)
            Predicted labels or values.
        
        Returns
        -------
        score : float
            Evaluation score.
        """
        pass
    
    @abstractmethod
    def get_default_metric(self):
        """Return the default metric name.
        
        Returns
        -------
        metric : str
            Default metric name.
        """
        pass











# Mô tả: Chứa class BaseEvaluator, định nghĩa giao diện cho các phương pháp đánh giá mô hình.
# Vai trò: Đảm bảo mọi evaluator có thể tính toán hiệu suất mô hình.

# Giải thích:
# BaseEvaluator yêu cầu hai phương thức:
# evaluate(): Tính điểm số (như accuracy, MSE) dựa trên nhãn thật và nhãn dự đoán.
# get_default_metric(): Trả về tên metric mặc định (như "accuracy" cho phân loại).
# Class này được kế thừa bởi các evaluator trong evaluators/ (như ClassificationEvaluator, RegressionEvaluator).
# Giao diện giúp AutoML so sánh các mô hình một cách nhất quán.