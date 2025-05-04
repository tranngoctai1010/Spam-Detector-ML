import argparse
import logging
from scripts.run_pipeline import run_pipeline
from scripts.train_model import main as train_model
from scripts.predict import predict  # Cần có file predict.py để xử lý dự đoán

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    filename="logs/main.log",
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    parser = argparse.ArgumentParser(description="Main entry point for the ML pipeline.")
    parser.add_argument("task", choices=["train", "run", "predict"], help="Task to execute")
    parser.add_argument("--input", type=str, help="Input text for prediction (required for 'predict')")

    args = parser.parse_args()
    logging.info(f"Starting task: {args.task}")

    if args.task == "train":
        print("Training the model...")
        train_model()
    elif args.task == "run":
        print("Running the entire pipeline...")
        run_pipeline()
    elif args.task == "predict":
        if not args.input:
            print("Error: Please provide input text using --input for prediction.") 
            exit(1)
        print(f"🔮 Making predictions for: {args.input}")
        result = predict(args.input)
        print(f"Prediction: {result}")

if __name__ == "__main__":
    main()



# # main.py
# import sys
# from scripts.run_pipeline import run_pipeline
# from scripts.train_model import main as train_model

# def main():
#     """
#     Entry point for the program.
#     """
#     # Check command-line arguments
#     if len(sys.argv) < 2:
#         print("Please provide a task. Example: python main.py <task>")
#         sys.exit(1)

#     task = sys.argv[1]

#     # If the task is 'train', only train the model
#     if task == 'train':
#         print("Training the model...")
#         train_model()

#     # If the task is 'run', run the entire pipeline
#     elif task == 'run':
#         print("Running the entire pipeline...")
#         run_pipeline()

#     else:
#         print(f"Task '{task}' is invalid. Please choose 'train' or 'run'.")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()











import argparse
from flask import Flask
from app.routes import create_app
from scripts.train_emails_model import train_email_classifier
from scripts.train_sms_model import train_sms_classifier

def main():
    parser = argparse.ArgumentParser(description="Email & SMS Spam Detector")
    parser.add_argument("--train", choices=["email", "sms", "all"], help="Train a model: email, sms, or all")
    parser.add_argument("--run", action="store_true", help="Run the API server")

    args = parser.parse_args()

    if args.train:
        if args.train == "email":
            print("Training email spam classifier...")
            train_email_classifier()
        elif args.train == "sms":
            print("Training SMS spam classifier...")
            train_sms_classifier()
        elif args.train == "all":
            print("Training both email and SMS spam classifiers...")
            train_email_classifier()
            train_sms_classifier()

    if args.run:
        app = create_app()
        app.run(debug=True, host="0.0.0.0", port=5000)

if __name__ == "__main__":
    main()
