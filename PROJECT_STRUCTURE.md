/Project
├── main.py
|
├── data/
|
├── configs/
|   ├── scripts_config.yaml
|   ├── logging_config.yaml
|   ├── processing_data_config.yaml
|   ├── training_model_config.yaml
|
├── logs/
|   ├── scripts.log
|   ├── src.log
|
├── scripts/
|   ├── evaluate_pipeline.py
|   ├── predict_pipeline.py
|   ├── train_pipeline.py
|   
├── src/
|   ├── processing_data/
|   |   ├── process_emails.py
|   ├── models/
|   |   ├──
|   ├── training_model/
|   |   ├── base_trainer.py
|   |   ├── classification.py
|   |   ├── regression.py
|   ├── evaluating_model/
|   |   ├── 
|   ├── utils/
|   |   ├── config_loader.py
|   |   ├── logger_manager.py
|   |   ├── model_handler.py
|   
├── tests/
|   ├── unit/
|   |   ├── test_src/
|   |   |   ├── test_utils
|   |   |   |   ├── test_logger_manager.py
|   |   |   |   ├── test_model_handler.py





2️⃣ Học MLOps (Docker, FastAPI, CI/CD) → Đưa mô hình AI vào thực tế.
3️⃣ Tối ưu code, tránh lỗi nhỏ (refactor preidct() thành predict(), tối ưu BaseTrainer).