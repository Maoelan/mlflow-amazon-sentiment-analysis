import mlflow
import mlflow.sklearn
import dagshub
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from xgboost import XGBClassifier
import pandas as pd
from preprocessing.amazon_preprocessing import load_and_preprocess_data
import time
from sklearn.metrics import roc_auc_score, log_loss
import os
import json

dagshub.init(repo_owner='Maoelan', repo_name='amazon-sentiment-analysis', mlflow=True)

# mlflow.set_tracking_uri("https://dagshub.com/Maoelan/amazon-sentiment-analysis.mlflow")  # use to Dagshub
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # use to local
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

param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

model = XGBClassifier(eval_metric='mlogloss')
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20, cv=5, verbose=2, random_state=42, n_jobs=-1)

# Start MLflow run
with mlflow.start_run():
    start_time = time.time()

    random_search.fit(X_train_tfidf, y_train)
    training_time = time.time() - start_time

    for param, value in random_search.best_params_.items():
        mlflow.log_param(param, value)

    mlflow.log_param("test_size", 0.3)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("vectorizer_max_features", 5000)
    mlflow.log_param("num_iter_random_search", 20)
    mlflow.log_param("cross_validation", 5)
    mlflow.log_param("num_parameters", json.dumps(random_search.best_estimator_.get_params()))

    best_model = random_search.best_estimator_

    y_pred = best_model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", report["weighted avg"]["precision"])
    mlflow.log_metric("recall", report["weighted avg"]["recall"])
    mlflow.log_metric("f1-score", report["weighted avg"]["f1-score"])
    mlflow.log_metric("train_score", best_model.score(X_train_tfidf, y_train))
    mlflow.log_metric("test_score", best_model.score(X_test_tfidf, y_test))

    mlflow.log_metric("roc_auc", roc_auc_score(y_test, best_model.predict_proba(X_test_tfidf), multi_class='ovr'))
    mlflow.log_metric("log_loss", log_loss(y_test, best_model.predict_proba(X_test_tfidf)))

    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
    mlflow.log_artifact('tfidf_vectorizer.pkl')

    joblib.dump(best_model, 'best_model.pkl')
    mlflow.log_artifact('best_model.pkl')

    model_size = os.path.getsize('best_model.pkl')
    model_size_mb = model_size / (1024 * 1024)
    mlflow.log_param("model_size_mb", model_size_mb)

    mlflow.log_metric("training_time", training_time)

    mlflow.sklearn.log_model(sk_model=best_model,
        artifact_path="model",
        input_example=input_example,
        signature=None
    )