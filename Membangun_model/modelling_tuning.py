import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from amazon_preprocessing import load_and_preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import git

dagshub.init(repo_owner='Maoelan', repo_name='amazon-sentiment-analysis', mlflow=True)

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Amazon Sentiment Analysis Model")

file_path = "amazon_reviews.csv"
data = load_and_preprocess_data(file_path)

X = data['content']
y = data['satisfaction_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

input_example = X_train_tfidf[0:5]

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
}

model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

with mlflow.start_run():
    grid_search.fit(X_train_tfidf, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, output_dict=True)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", report["weighted avg"]["precision"])
    mlflow.log_metric("recall", report["weighted avg"]["recall"])
    mlflow.log_metric("f1-score", report["weighted avg"]["f1-score"])

    mlflow.sklearn.log_model(sk_model=best_model,
        artifact_path="model",
        input_example=input_example,
        signature=None
    )

    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
    mlflow.log_artifact('tfidf_vectorizer.pkl')

    repo = git.Repo(search_parent_directories=True)
    repo.git.add(A=True)
    repo.index.commit("Training completed with best model")
    repo.remotes.origin.push()