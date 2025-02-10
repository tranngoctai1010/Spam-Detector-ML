from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import LogisticRegression

def build_pipeline():
    pipeline = Pipeline(steps=[
        ("vetorizer", TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=10000))
        ("feature_selector", SelectPercentile(chi2, percentile=30))
    ])
    
