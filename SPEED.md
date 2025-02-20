/project
├── benchmarks/
│   ├── local_tests/               # Đo trên máy cá nhân
│   │   ├── test_latency.py        # Đo thời gian xử lý trên máy cá nhân
│   │   ├── test_resource_usage.py # Đo CPU, RAM trên máy cá nhân
│   │   ├── test_throughput.py     # Đo throughput trên máy cá nhân
│   │   ├── README.md              # Hướng dẫn chạy test trên máy cá nhân
│   │
│   ├── server_tests/              # Đo trên server
│   │   ├── test_latency.py        # Đo thời gian xử lý trên server
│   │   ├── test_resource_usage.py # Đo CPU, RAM trên server
│   │   ├── test_throughput.py     # Đo throughput trên server
│   │   ├── setup_monitoring.md    # Hướng dẫn cài đặt monitoring trên server (Prometheus + Grafana)
│   │   ├── README.md              # Hướng dẫn chạy test trên server




/project
├── evaluation/                 # Thư mục đánh giá mô hình
│   ├── baseline.py             # Tạo baseline để so sánh
│   ├── benchmark.py            # So sánh hiệu suất các mô hình
│   ├── results/                # Lưu kết quả đánh giá
│   │   ├── benchmark_results.csv
│   │   ├── baseline_results.csv
│   ├── metrics.py              # Hàm đánh giá (accuracy, precision, recall...)
│   ├── visualize.py            # Vẽ biểu đồ so sánh



Monitoring trong AI bao gồm:
Theo dõi hiệu suất mô hình:

Độ chính xác (accuracy), độ lỗi (loss), precision, recall, F1-score…
So sánh với baseline để xem có bị suy giảm không (model drift).
Theo dõi tốc độ và tài nguyên hệ thống:

Thời gian phản hồi của API (latency).
Sử dụng CPU, GPU, RAM, ổ cứng.
Mức tiêu thụ điện năng (nếu quan trọng).
Theo dõi dữ liệu đầu vào:

Kiểm tra dữ liệu có bị lệch phân phối so với dữ liệu train không.
Phát hiện dữ liệu outlier hoặc bất thường.
Cách triển khai Monitoring:
Log hệ thống (app.log, error.log, pipeline.log).
Prometheus + Grafana: Theo dõi hiệu suất mô hình trong thực tế.
TensorBoard: Theo dõi quá trình train model.
