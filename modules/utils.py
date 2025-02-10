import joblib

def save_model(pipeline, model_path="models/spam_classifier.pkl"):
    joblib.dump(pipeline, model_path)
