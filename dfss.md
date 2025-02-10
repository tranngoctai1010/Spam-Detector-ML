### File: modules/__init__.py

from .preprocess import load_and_clean_data, encode_labels, split_data
from .train import build_pipeline, train_model, save_model
from .evaluate import evaluate_model

__all__ = [
    "load_and_clean_data",
    "encode_labels",
    "split_data",
    "build_pipeline",
    "train_model",
    "save_model",
    "evaluate_model"
]


### File: modules/preprocess.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(filepath):
    data = pd.read_csv(filepath)
    data = data.dropna()
    return data

def encode_labels(data, target_column="label"):
    labelencoder = LabelEncoder()
    data[target_column] = labelencoder.fit_transform(data[target_column])
    return data, labelencoder

def split_data(data, text_column="text", target_column="label", test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    x = data[text_column]
    y = data[target_column]
    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)


### File: modules/train.py

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_modl import LogisticRegression
import joblib

def build_pipeline():
    return Pipeline(steps=[
        ("vectorizer", TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=10000)),
        ("feature_selector", SelectPercentile(chi2, percentile=50)),
        ("model", LogisticRegression(max_iter=1000))
    ])

def train_model(pipeline, x_train, y_train):
    pipeline.fit(x_train, y_train)
    return pipeline


def save_model(pipeline, model_path="models/spam_classifier.pkl"):
    joblib.dump(pipeline, model_path)


### File: modules/evaluate.py

from sklearn.metrics import classification_report

def evaluate_model(pipeline, x_test, y_test):
    y_pred = pipeline.predict(x_test)
    report = classification_report(y_test, y_pred)
    print(report)
    return report


### File: scripts/run_pipeline.py

from modules import load_and_clean_data, encode_labels, split_data, build_pipeline, train_model, save_model, evaluate_model

# Load and preprocess data
data = load_and_clean_data("data/spam_Emails_data.csv")
data, labelencoder = encode_labels(data)

# Split data
x_train, x_test, y_train, y_test = split_data(data)

# Build and train model
pipeline = build_pipeline()
pipeline = train_model(pipeline, x_train, y_train)

# Evaluate model
evaluate_model(pipeline, x_test, y_test)

# Save model
save_model(pipeline)


### File: requirements.txt
pandas
scikit-learn
joblib


### File: README.md
# Spam Email Classifier

This project builds a spam email classifier using Logistic Regression and TF-IDF vectorization.

## Project Structure

```
/project
├── main.py                  # Run the full system
├── requirements.txt         # Required libraries
├── README.md                # Project documentation
├── data/                    # Dataset folder
│   └── spam_Emails_data.csv # Spam dataset
├── models/                  # Saved trained models
│   └── spam_classifier.pkl  # Trained model
├── modules/                 # Logic modules
│   ├── preprocess.py        # Data preprocessing
│   ├── train.py             # Model training
│   ├── evaluate.py          # Model evaluation
├── scripts/                 # Execution scripts
│   └── run_pipeline.py      # Full pipeline execution
```

## How to Run

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Run the full pipeline:
```
python scripts/run_pipeline.py
```

---

Vậy là bạn đã chia nhỏ code thành các module rõ ràng để dễ quản lý và mở rộng sau này!
