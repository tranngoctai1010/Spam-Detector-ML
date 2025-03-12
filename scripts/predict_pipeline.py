# Build-in imports
import re

# Internal imports 
from src.utils.model_handler import ModelHandler

# Handle input email
def handle_email_input(text) -> str:
    text = text.lower()
    text = re.sub(r"\W+", "", text)
    return text

def predict_email():
    pass
    
    












# import logging
# import yaml
# import re
# import pandas as pd
# from modules.utils import ModelHandler
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_selection import SelectPercentile, chi2

# # Config logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     filename="logs/scripts.log",
#     filemode='w',
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )

# # Load configuration file
# try:
#     file_path = "configs/scripts_config.yaml"
#     with open(file_path, "r") as file:
#         full_config = yaml.safe_load(file)
#         config = full_config["predict_emails.py"]
# except Exception as e:
#     logging.error("Error when reading scripts_config.yaml file in predict_emails.py file")
#     raise

# # Tiền xử lý email
# def clean_email(text):
#     text = text.lower()
#     text = re.sub(r'\W+', ' ', text)  # Loại bỏ ký tự đặc biệt
#     return text

# # Load dữ liệu mẫu để khớp đặc trưng đã chọn
# try:
#     train_data = pd.read_csv("datasets/emails.csv").dropna()
#     X_train_texts = train_data.drop("label", axis=1).values.ravel()  # Lấy nội dung email
# except Exception as e:
#     logging.error("Error loading training data for feature matching.")
#     raise

# # Khởi tạo vectorizer và bộ chọn đặc trưng giống lúc train
# vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=10000)
# selector = SelectPercentile(chi2, percentile=30)

# X_train_tfidf = vectorizer.fit_transform(X_train_texts)
# X_train_selected = selector.fit_transform(X_train_tfidf, train_data["label"])  # Áp dụng chọn đặc trưng

# def predict_emails(email):
#     model_handler = ModelHandler()
    
#     # Load model
#     logging.info("Loading model...")
#     model_handler.load_model(filename=config["load_model"])
#     model = model_handler.best_estimator
#     logging.info("Model loaded successfully.")
    
#     # Tiền xử lý email
#     email_cleaned = clean_email(email)
    
#     # Biến đổi email thành vector đặc trưng
#     email_tfidf = vectorizer.transform([email_cleaned])
#     email_selected = selector.transform(email_tfidf)  # Giữ lại 30% đặc trưng quan trọng nhất
    
#     # Dự đoán
#     result = model.predict(email_selected)
#     return result[0]

# if __name__ == "__main__":
#     sample_email = "Congratulations! You've won a free iPhone. Click here to claim your prize."
#     prediction = predict_emails(sample_email)
#     print(f"Prediction: {'Spam' if prediction == 1 else 'Not Spam'}")
