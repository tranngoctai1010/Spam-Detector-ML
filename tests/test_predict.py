# tests/test_predict.py
import unittest
from modules.training_modules.classification import TrainClassification
from modules.process_emails import process_emails

class TestPredict(unittest.TestCase):
    
    def test_predict(self):
        # Tiền xử lý dữ liệu
        x_train, x_test, y_train, y_test = process_emails()
        
        # Huấn luyện mô hình
        model = TrainClassification(x_train, x_test, y_train, y_test)
        model.train()
        
        # Dự đoán trên dữ liệu test
        predictions = model.best_estimator.predict(x_test)
        
        # Kiểm tra xem dự đoán có phải là một mảng
        self.assertTrue(isinstance(predictions, list))
        self.assertEqual(len(predictions), len(y_test))  # Kiểm tra số lượng dự đoán có bằng số lượng mẫu test không
        
if __name__ == "__main__":
    unittest.main()
