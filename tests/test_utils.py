# tests/test_utils.py
import unittest
from modules.utils import ModelHandler

class TestUtils(unittest.TestCase):
    
    def test_save_model(self):
        # Giả sử bạn có một mô hình giả
        model_handler = ModelHandler()
        
        # Kiểm thử lưu mô hình
        model_handler.save_model(model="dummy_model", filename="test_model.pkl")
        
        # Kiểm tra xem file đã được tạo chưa
        import os
        self.assertTrue(os.path.exists("test_model.pkl"))
        
if __name__ == "__main__":
    unittest.main()
