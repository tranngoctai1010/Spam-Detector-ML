from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    """Abstract base class for optimizers."""
    
    @abstractmethod
    def optimize(self, pipeline, param_grid, X, y):
        """Optimize the model with given parameters.
        
        Parameters
        ----------
        pipeline : object
            Scikit-learn pipeline or model to optimize.
        param_grid : dict
            Parameter grid to search.
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        best_estimator : object
            Best model instance.
        best_score : float
            Best score achieved.
        """
        pass




# Mô tả: Chứa class BaseOptimizer, định nghĩa giao diện cho các phương pháp tối ưu hóa siêu tham số (như GridSearchCV).
# Vai trò: Đảm bảo mọi optimizer có phương thức để tối ưu hóa mô hình với lưới siêu tham số.


# Giải thích:
# BaseOptimizer yêu cầu một phương thức optimize(), nhận pipeline (hoặc mô hình), lưới siêu tham số, và dữ liệu để tìm cấu hình tốt nhất.
# Class này được kế thừa bởi các optimizer trong optimizers/ (như GridSearchOptimizer, RandomSearchOptimizer).
# Giao diện cho phép AutoML gọi bất kỳ optimizer nào mà không cần biết chi tiết triển khai.