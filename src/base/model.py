from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for model wrappers."""
    
    @abstractmethod
    def get_model(self):
        """Return the scikit-learn model instance.

        Returns
        -------
        model : object
            Scikit-learn model instance.
        """
        pass

    @abstractmethod
    def get_param_grid(self):
        """Return the parameter grid for optimization.
        
        Returns
        -------
        param_grid : dict
            Dictionary of parameters to search.
        """
        pass






# Mô tả: Chứa class BaseModel, định nghĩa giao diện cho các wrapper mô hình (như LogisticRegression, RandomForest).
# Vai trò: Đảm bảo mỗi wrapper mô hình cung cấp instance mô hình và lưới siêu tham số cho tối ưu hóa.

# Giải thích:
# BaseModel yêu cầu hai phương thức trừu tượng:
# get_model(): Trả về instance của mô hình (như LogisticRegression()).
# get_param_grid(): Trả về dictionary chứa siêu tham số để dùng trong GridSearchCV (như {"model__C": [0.1, 1.0]}).
# Class này được kế thừa bởi các wrapper trong models/ (như LogisticRegressionModel, RandomForestModel).
# Giao diện đơn giản, chỉ tập trung vào chuẩn hóa cách AutoML truy cập mô hình.
# 3. base/optimizer.py