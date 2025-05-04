import joblib
from amazon_preprocessing_inference import preprocessing_text, transform_text_to_tfidf
import requests

def load_tfidf(tfidf_path):
    return joblib.load(tfidf_path)

def preprocess_and_vectorize(text, tfidf_vectorizer):
    processed = preprocessing_text(text)
    vectorized = transform_text_to_tfidf([processed], tfidf_vectorizer)
    return vectorized.toarray().tolist()

def send_prediction_to_server(vector_input):
    url = "http://127.0.0.1:8000/predict"
    headers = {"Content-Type": "application/json"}
    data = {"instances": vector_input}

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def predict(text_data):
    tfidf_path = 'tuning_xgboost_tfidf_vectorizer.joblib'
    tfidf_vectorizer = load_tfidf(tfidf_path)

    vector_input = preprocess_and_vectorize(text_data[0], tfidf_vectorizer)

    result = send_prediction_to_server(vector_input)
    
    if result:
        sentiment_labels = {0: "Dissatisfied", 1: "Neutral", 2: "Satisfied"}
        predicted_class = result.get('predictions')[0]
        sentiment = sentiment_labels.get(predicted_class, "Unknown")
        return sentiment
    else:
        return "Error in prediction"

def inference():
    user_input = input("Enter a review for sentiment prediction: ")
    sentiment = predict([user_input])
    print(f"Sentiment Prediction: {sentiment}")

if __name__ == "__main__":
    inference()