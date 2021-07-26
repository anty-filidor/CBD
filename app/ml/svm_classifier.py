from typing import List

import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle


def do_nothing(x):
    return x


def init_model():
    return Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=do_nothing, lowercase=False)),
            ("tfidf", TfidfTransformer()),
            ("clf", LinearSVC(class_weight="balanced")),
        ]
    )


def train(dataset, model, model_path):
    X, y = dataset.df["tokens"].values, dataset.df["tag"].values
    X, y = shuffle(X, y, random_state=42)
    train_x, valid_x, train_y, valid_y = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    model.fit(train_x, train_y)

    with open(model_path, "wb") as file:
        joblib.dump(model, file)

    y_pred = model.predict(valid_x)
    return accuracy_score(valid_y, y_pred), f1_score(valid_y, y_pred, average="macro")


def tag_data(tokenized_data: List[List[str]], model):
    return [model.predict([item])[0] if item else 0 for item in tokenized_data]
