# tests/test_preprocess.py
import unittest
from modules.process_emails import process_emails

class TestPreprocess(unittest.TestCase):
    
    def test_process_emails(self):
        # Giả sử process_emails() trả về 4 giá trị
        x_train, x_test, y_train, y_test = process_emails()
        
        # Kiểm tra xem dữ liệu có được phân chia chính xác
        self.assertEqual(len(x_train), 1000)  # Ví dụ
        self.assertEqual(len(y_train), 1000)  # Ví dụ
        self.assertTrue(isinstance(x_train, list))  # Kiểm tra kiểu dữ liệu
        
if __name__ == "__main__":
    unittest.main()
