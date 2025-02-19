from modules.utils import ModelHandler
import logging
import yaml

#Config logging
logging.basicConfig(
    level=logging.DEBUG,
    filename="logs/scripts.log",
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#Load configuration file
try:
    file_path = "configs/scripts_config.yaml"
    with open(file_path, "r") as file:
        full_config = yaml.safe_load(file)
        config = full_config["predict_emails.py"]
except Exception as e:
    logging.error("Error when reading scripts_config.yaml file in predict_emails.py file")
    raise


def predict_emails(email):
    
    model_handler = ModelHandler()

    #Step1: Load model
    logging.info("Starting to load model .....")
    model_handler.load_model(filename=config["load_model"])
    model = model_handler.best_estimator
    logging.info("Model load successfully.")
    
    #Step: Predict email
    result = model.predict([email])
    
    
    
    

# import re

# def clean_email(text):
#     """Tiền xử lý email (loại bỏ ký tự đặc biệt, viết thường)"""
#     text = text.lower()
#     text = re.sub(r'\W+', ' ', text)  # Loại bỏ ký tự đặc biệt
#     return text

# def extract_features(email_text):
#     """Chuyển đổi email thành vector đặc trưng (giả lập)"""
#     words = email_text.split()
#     return [len(words), sum(len(word) for word in words)]
