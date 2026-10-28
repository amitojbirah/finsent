from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_dummy():  
    return DummyClassifier(strategy="most_frequent")

def build_tfidf_lr(class_weight=None):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,sublinear_tf=True, )),
        ("lr", LogisticRegression(class_weight=class_weight, max_iter=1000))
    ])
    return pipeline
