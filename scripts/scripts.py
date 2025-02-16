import pandas as pd
from modules.preprocess import load_and_clean_data, encode_target, split_data
from modules.train_models.classification import TrainClassification
import joblib

def preprocess():
    input_file = "datasets/emails.csv"
    output_file = "data/processed/cleaned_emails.csv"
    
    print("🔄 Loading and cleaning data...")
    data = load_and_clean_data(input_file)
    
    print("🔄 Encoding target column...")
    data = encode_target(data, "label")
    
    print("💾 Saving processed data...")
    data.to_csv(output_file, index=False)
    print(f"✅ Processed data saved to {output_file}")

def train_model():
    print("🚀 Training model...")
    data = pd.read_csv("data/processed/cleaned_emails.csv")
    x_train, x_test, y_train, y_test = split_data(data, "label")
    
    trainer = TrainClassification(x_train, x_test, y_train, y_test)
    trainer.train()
    trainer.evaluate()
    
    model_path = "models/spam_classifier.pkl"
    joblib.dump(trainer.best_estimator, model_path)
    print(f"✅ Model saved to {model_path}")

def predict_message(text):
    print("🔍 Loading model for prediction...")
    model_path = "models/spam_classifier.pkl"
    model = joblib.load(model_path)
    prediction = model.predict([text])[0]
    label = "Spam" if prediction == 1 else "Ham"
    print(f"📩 Predicted label: {label}")
    return label

def run_pipeline():
    preprocess()
    train_model()
    sample_text = "Win a brand new car! Click here now!"
    predict_message(sample_text)

if __name__ == "__main__":
    run_pipeline()
