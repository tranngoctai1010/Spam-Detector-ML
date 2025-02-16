import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
import logging

#Config logging
logging.basicConfig(
    level=logging.DEBUG,
    filename="logs/dev.log",
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_emails():
    """
    This function processes data for the emails.csv file.        
    """
    try:
        data = pd.read_csv("datasets/emails.csv")
        data = data.dropna()

        target = "label"
        x = data.drop(target, axis=1)
        y = data[target]

        labelencoder = LabelEncoder()
        y = labelencoder.fit_transform(y)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

        pipeline = Pipeline(steps=[
            ("vectorizer", TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=10000)),
            ("select_feature", SelectPercentile(chi2, percentile=30)),
            ("classifier", MultinomialNB())
        ])

        pipeline.fit(x_train, y_train)
    except Exception as e:
        logging.error("Error when processing emails.csv file.")
        raise
    return x_train, x_test, y_train, y_test