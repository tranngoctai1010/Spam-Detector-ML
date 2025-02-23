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



👨‍💻 AI Engineer – Xây model, tối ưu, huấn luyện.
👨‍💻 MLOps Engineer – Lo phần deployment, pipeline, monitoring.
👨‍💻 Backend Engineer – Lo API, database, logic hệ thống.
👨‍💻 Frontend Engineer – Nếu có UI/UX thì cần luôn.
👨‍💻 DevOps Engineer – Lo hạ tầng, scaling, CI/CD.
📊 Data Engineer – Xử lý pipeline, data preprocessing.



🚀 API backend để gọi model.
🚀 Benchmarking để đo hiệu suất.
🚀 Monitoring để giám sát khi chạy thực tế.
🚀 Tự động hóa pipeline AI.