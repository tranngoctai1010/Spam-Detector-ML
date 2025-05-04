from abc import ABC, abstractmethod

class BaseAutoML(ABC):
    """Abstract base class for AutoML systems."""
    
    @abstractmethod
    def fit(self, X, y):
        """Fit the AutoML system on training data.
        
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
        """Predict using the best model.
        
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
    
    @abstractmethod
    def get_best_model(self):
        """Return the best trained model.
    
        Returns
        -------
        model : object
            Best model instance.
        """
        pass







# Mô tả: Chứa class BaseAutoML, định nghĩa giao diện cho class AutoML chính, là trung tâm điều phối của module.
# Vai trò: Đảm bảo class AutoML có các phương thức cơ bản như huấn luyện (fit), dự đoán (predict), và lấy mô hình tốt nhất (get_best_model).


# Giải thích:
# BaseAutoML định nghĩa ba phương thức trừu tượng bắt buộc:
# fit(): Huấn luyện AutoML trên dữ liệu X, y.
# predict(): Dự đoán trên dữ liệu mới X bằng mô hình tốt nhất.
# get_best_model(): Trả về mô hình tốt nhất sau khi tối ưu hóa.
# Docstring theo chuẩn scikit-learn, mô tả rõ tham số và giá trị trả về.
# Class này sẽ được kế thừa bởi AutoML trong auto_ml.py, nơi logic cụ thể (thử mô hình, tối ưu hóa) được triển khai.



    