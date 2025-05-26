import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler, SMOTEN

def filter_location(location):
    result = re.findall("\,\s[A-Z]{2}$", location)
    if len(result) != 0:
        return result[0][2:]
    else:
        return location

class JobPostProcessor:

    @staticmethod
    def process():
        data = pd.read_excel("final_project.ods", engine="odf", dtype=str)
        data = data.dropna(axis=0)
        data["location"] = data["location"].apply(filter_location)

        target = "career_level"
        x = data.drop(target, axis=1)
        y = data[target]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100, stratify=y)

        preprocessor = ColumnTransformer(transformers=[
            ("title_ft", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "title"),
            ("location_ft", OneHotEncoder(handle_unknown="ignore"), ["location"]),
            ("des_ft", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.01, max_df=0.95), "description"),
            ("function_ft", OneHotEncoder(handle_unknown="ignore"), ["function"]),
            ("industry_ft", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "industry")
        ])

        pipeline = [
            ("preprocessor", preprocessor),   # (6458, 7990)
            ("feature_selector", SelectPercentile(chi2, percentile=5)),
        ]

        params = {
            "feature_selector__percentile": [1, 5, 10]
        }
        
        return x_train, x_test, y_train, y_test, pipeline, params

