import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_clean_data(file_path):

    data = pd.read_csv(file_path)

    data = data.dropna()

    return data 

def encode_target(data, target_column):

    labelencoder = LabelEncoder()

    data[target_column] = labelencoder.fit_transform(data[target_column])

    return data

def split_data(data, target):
    
    x = data.drop(target, axis=1)
    y = data[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42, stratify=y)

    return x_train, x_test, y_train, y_test




import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(file_path):
    """Đọc dữ liệu từ file CSV"""
    return pd.read_csv(file_path)

def clean_data(df):
    """Làm sạch dữ liệu: xóa NaN, loại bỏ trùng lặp"""
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def encode_categorical(df, categorical_columns):
    """Mã hóa các cột dạng phân loại (Category) sang số"""
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

def scale_data(df, numerical_columns):
    """Chuẩn hóa dữ liệu số (đưa về cùng khoảng giá trị)"""
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df, scaler

def preprocess_pipeline(file_path, categorical_columns, numerical_columns):
    """Pipeline tổng hợp cho tất cả các bước tiền xử lý"""
    df = load_data(file_path)
    df = clean_data(df)
    df, encoders = encode_categorical(df, categorical_columns)
    df, scaler = scale_data(df, numerical_columns)
    return df, encoders, scaler
