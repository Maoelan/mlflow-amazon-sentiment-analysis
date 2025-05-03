import mlflow
import mlflow.sklearn
import joblib
import time
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, log_loss, confusion_matrix
from xgboost import XGBClassifier
from preprocessing.amazon_preprocessing import load_and_preprocess_data

# mlflow.set_tracking_uri("https://dagshub.com/Maoelan/amazon-sentiment-analysis.mlflow") # use dagshub
mlflow.set_tracking_uri("http://127.0.0.1:5000") # using local MLflow server
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

xgb_model = XGBClassifier(eval_metric='mlogloss')

artifact_folder = "Saved_Artifacts_No_Tuning"
os.makedirs(artifact_folder, exist_ok=True)

with mlflow.start_run():
    start_time = time.time()

    xgb_model.fit(X_train_tfidf, y_train)
    training_time = time.time() - start_time

    mlflow.log_param("test_size", 0.3)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("vectorizer_max_features", 5000)

    y_pred = xgb_model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", report["weighted avg"]["precision"])
    mlflow.log_metric("recall", report["weighted avg"]["recall"])
    mlflow.log_metric("f1-score", report["weighted avg"]["f1-score"])
    mlflow.log_metric("train_score", xgb_model.score(X_train_tfidf, y_train))
    mlflow.log_metric("test_score", xgb_model.score(X_test_tfidf, y_test))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, xgb_model.predict_proba(X_test_tfidf), multi_class='ovr'))
    mlflow.log_metric("log_loss", log_loss(y_test, xgb_model.predict_proba(X_test_tfidf)))

    model_filename = f"{artifact_folder}/xgboost_model.pkl"
    joblib.dump(xgb_model, model_filename)
    mlflow.log_artifact(model_filename)

    tfidf_filename = f"{artifact_folder}/xgboost_tfidf_vectorizer.pkl"
    joblib.dump(tfidf, tfidf_filename)
    mlflow.log_artifact(tfidf_filename)

    model_size = os.path.getsize(model_filename)
    model_size_mb = model_size / (1024 * 1024)
    mlflow.log_param("model_size_mb", model_size_mb)

    mlflow.log_metric("training_time", training_time)

    metric_info = {
        "accuracy": accuracy,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1-score": report["weighted avg"]["f1-score"],
        "train_score": xgb_model.score(X_train_tfidf, y_train),
        "test_score": xgb_model.score(X_test_tfidf, y_test),
        "roc_auc": roc_auc_score(y_test, xgb_model.predict_proba(X_test_tfidf), multi_class='ovr'),
        "log_loss": log_loss(y_test, xgb_model.predict_proba(X_test_tfidf))
    }

    metric_filename = f"{artifact_folder}/metric_info.json"
    with open(metric_filename, 'w') as f:
        json.dump(metric_info, f)
    mlflow.log_artifact(metric_filename)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = range(len(set(y)))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    cm_filename = f"{artifact_folder}/training_cm.png"
    plt.savefig(cm_filename)
    mlflow.log_artifact(cm_filename)

    mlflow.sklearn.log_model(
        sk_model=xgb_model,
        artifact_path="model",
        input_example=input_example,
        signature=None
    )