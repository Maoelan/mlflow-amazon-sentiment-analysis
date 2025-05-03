import mlflow
import mlflow.sklearn
import dagshub
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from xgboost import XGBClassifier
import pandas as pd
from preprocessing.amazon_preprocessing import load_and_preprocess_data

dagshub.init(repo_owner='Maoelan', repo_name='amazon-sentiment-analysis', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/Maoelan/amazon-sentiment-analysis.mlflow")
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

model = XGBClassifier(eval_metric='mlogloss')
with mlflow.start_run():
    model.fit(X_train_tfidf, y_train)

    mlflow.log_param("vectorizer_max_features", 5000)
    mlflow.log_param("test_size", 0.3)
    mlflow.log_param("random_state", 42)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", report["weighted avg"]["precision"])
    mlflow.log_metric("recall", report["weighted avg"]["recall"])
    mlflow.log_metric("f1-score", report["weighted avg"]["f1-score"])

    mlflow.sklearn.log_model(sk_model=model,
                             artifact_path="model",
                             input_example=input_example,
                             signature=None)

    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
    mlflow.log_artifact('tfidf_vectorizer.pkl')

    joblib.dump(model, 'best_model.pkl')
    mlflow.log_artifact('best_model.pkl')