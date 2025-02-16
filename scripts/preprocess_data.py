import argparse
import pandas as pd
import yaml
from modules.preprocess import load_and_clean_data, encode_target, split_data

# Load config file
with open("configs/dev_config.yaml", "r") as file:
    config = yaml.safe_load(file)["preprocessing"]

def preprocess():
    """Tiền xử lý dữ liệu và lưu vào file"""
    input_file = config["input_file"]
    output_file = config["output_file"]

    print("🔄 Loading and cleaning data...")
    data = load_and_clean_data(input_file)

    print("🔄 Encoding target column...")
    data = encode_target(data, "label")

    print("✂ Splitting data into train & test...")
    x_train, x_test, y_train, y_test = split_data(data, "label")

    print("💾 Saving processed data...")
    data.to_csv(output_file, index=False)
    x_train.to_csv(config["train_file"], index=False)
    x_test.to_csv(config["test_file"], index=False)
    y_train.to_csv(config["label_train"], index=False)
    y_test.to_csv(config["label_test"], index=False)

    print(f"✅ Processed data saved to {output_file}")

if __name__ == "__main__":
    preprocess()
