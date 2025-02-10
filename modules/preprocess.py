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