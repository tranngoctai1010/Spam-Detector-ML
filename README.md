/project
├── venv/                   # Môi trường ảo
├── main.py
├── requirements.txt
├── README.md
├── data/
├── models/
├── modules/
├── notebooks/
├── scripts/
├── configs/
├── tests/



Lưu mô hình trong modules/predict_response.py.
Tạo file modules/use_model.py để dự đoán.
Tạo file scripts/run_pipeline.py để chạy toàn bộ quy trình.
Viết kiểm thử trong tests/test_predict_response.py.
Cập nhật tài liệu trong README.md.


/project
├── main.py                  # Chạy toàn bộ hệ thống
├── requirements.txt         # Các thư viện cần thiết
├── README.md                # Tài liệu dự án
├── data/                    # Dữ liệu
│   ├── sms_spam.csv         # Dataset SMS Spam
├── models/                  # Lưu trữ mô hình
│   ├── spam_classifier.pkl  # Mô hình huấn luyện xong
├── modules/                 # Các module xử lý logic
│   ├── __init__.py
│   ├── preprocess.py        # Xử lý dữ liệu văn bản
│   ├── train.py             # Huấn luyện mô hình
│   ├── predict.py           # Dự đoán tin nhắn mới 
│   ├── evaluate.py          # Đánh giá mô hình
|   ├── utils.py             # Các hàm tiện ích dùng chung
├── notebooks/               # Phân tích và thử nghiệm
│   ├── sms_analysis.ipynb   # Phân tích và huấn luyện thử
├── scripts/                 # Các script thực thi
│   ├── run_pipeline.py      # Chạy pipeline từ đầu đến cuối
│   ├── preprocess_data.py   # Xử lý dữ liệu độc lập
│   ├── train_model.py       # Huấn luyện mô hình độc lập
│   ├── predict_message.py   # Dự đoán tin nhắn độc lập
├── configs/                 # Cấu hình dự án
│   ├── pipeline_config.yaml # Cấu hình xử lý
├── tests/                   # Kiểm thử
│   ├── test_preprocess.py   # Kiểm thử xử lý dữ liệu
│   ├── test_predict.py      # Kiểm thử dự đoán
│   ├── test_train.py        # Kiểm thử huấn luyện
