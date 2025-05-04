# Cài đặt môi trường
install:
	pip install -r requirements.txt

# Train model
train:
	python train.py

# Test model
test:
	python test.py

# Format code bằng black
format:
	black .

# Dọn rác .pyc, __pycache__
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -r {} +

# Chạy tất cả (giống như pipeline nhỏ)
all: install format train test
