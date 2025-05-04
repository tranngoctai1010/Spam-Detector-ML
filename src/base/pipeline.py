from abc import ABC, abstractmethod

class BasePipeline(ABC):
    """Abstract base class for pipelines."""
    
    @abstractmethod
    def fit(self, X, y):
        """Fit the pipeline on training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : object
            Returns self.
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """Predict using the pipeline.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict.
        
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values.
        """
        pass








# Mô tả: Chứa class BasePipeline, định nghĩa giao diện cho các pipeline (kết hợp tiền xử lý, mô hình, v.v.).
# Vai trò: Đảm bảo mọi pipeline có thể huấn luyện và dự đoán.

# Giải thích:
# BasePipeline yêu cầu hai phương thức:
# fit(): Huấn luyện pipeline trên dữ liệu.
# predict(): Dự đoán trên dữ liệu mới.
# Class này được kế thừa bởi các pipeline trong pipelines/ (như BasicPipeline, FeaturePipeline).
# Giao diện đơn giản, hỗ trợ AutoML sử dụng pipeline một cách thống nhất.