import pandas as pd
from modules.preprocess import load_and_clean_data, encode_target, split_data

def main():
    input_file = "datasets/emails.csv"
    output_file = "data/processed/cleaned_emails.csv"
    
    print("🔄 Loading and cleaning data...")
    data = load_and_clean_data(input_file)
    
    print("🔄 Encoding target column...")
    data = encode_target(data, "label")
    
    print("💾 Saving processed data...")
    data.to_csv(output_file, index=False)
    print(f"✅ Processed data saved to {output_file}")

if __name__ == "__main__":
    main()
