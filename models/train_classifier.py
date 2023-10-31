import sys
import pandas as pd

import nltk
import pickle
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from typing import List

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")


def load_data(database_filepath):
    """load disaster meesages and categories from sqlite database

    Parameters
    ----------
    database_filepath : str

    Returns
    -------
    Tuple[X,Y,category_names]
    """
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table("DisasterMessagesTable", con=engine)

    X = df["message"]
    Y = df.drop(columns=["message", "original", "genre"])
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text: str) -> List[str]:
    """toeknize function to prepare message data for MLP

    Parameters
    ----------
    text : str

    Returns
    -------
    List[str]
    """
    url_regex = (
        "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    tokens = word_tokenize(
        text,
        language="english",
    )
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]

    removed_stop_words = [
        tok for tok in clean_tokens if tok not in stopwords.words("english")
    ]

    return removed_stop_words


def build_model():
    """
    Returns a GridSearchCV object, with the pipeline to apply to the input
    data, and the parameters to tune on the training data to find the "best"
    model
    """
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(RandomForestClassifier())),
        ]
    )
    parameters = {
        "clf__estimator__n_estimators": [5, 10],
        "clf__estimator__min_samples_split": [2, 3],
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Print key metrics evaluating model fit for each category column"""
    y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print(f"For {category_names[i]}:")
        print(classification_report(Y_test.iloc[:, i], y_pred[:, i]))
        print("-" * 50)


def save_model(model, model_filepath):
    """save best model in a pickle object"""
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)


def main():
    """Run the script to load data, train model and save it"""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model.best_estimator_, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model.best_estimator_, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
