import joblib
from amazon_preprocessing_inference import preprocessing_text, transform_text_to_tfidf
import requests

sentiment_labels = {0: "Dissatisfied", 1: "Neutral", 2: "Satisfied"}

def load_tfidf(tfidf_path):
    return joblib.load(tfidf_path)

def preprocess_and_vectorize(text, tfidf_vectorizer):
    processed = preprocessing_text(text)
    vectorized = transform_text_to_tfidf([processed], tfidf_vectorizer)
    return vectorized.toarray().tolist()

def send_prediction_to_server(vector_input):
    url = "http://127.0.0.1:8080/invocations"
    headers = {"Content-Type": "application/json"}
    data = {"instances": vector_input}

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def single_inference():
    tfidf_path = 'tuning_xgboost_tfidf_vectorizer.pkl'
    tfidf_vectorizer = load_tfidf(tfidf_path)

    user_input = input("Enter a review for sentiment prediction: ")
    vector_input = preprocess_and_vectorize(user_input, tfidf_vectorizer)

    result = send_prediction_to_server(vector_input)
    if result:
        predicted_class = result.get('predictions')[0]
        sentiment = sentiment_labels.get(predicted_class, "Unknown")
        print(f"Sentiment Prediction: {sentiment}")

if __name__ == "__main__":
    single_inference()