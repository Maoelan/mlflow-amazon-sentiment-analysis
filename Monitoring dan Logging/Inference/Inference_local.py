import joblib
from amazon_preprocessing_inference import preprocessing_text, transform_text_to_tfidf

def load_model_and_tfidf(model_path, tfidf_path):
    model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(tfidf_path)
    return model, tfidf_vectorizer

def predict(text_data, model, tfidf_vectorizer):
    processed_text = [preprocessing_text(text) for text in text_data]

    text_tfidf = transform_text_to_tfidf(processed_text, tfidf_vectorizer)

    predictions = model.predict(text_tfidf)

    sentiment_labels = {0: "Dissatisfied", 1: "Neutral", 2: "Satisfied"}
    categorical_prediction = [sentiment_labels.get(pred, "Unknown") for pred in predictions]
    
    return categorical_prediction

def inference():
    model_path = 'tuning_xgboost_best_model.joblib'
    tfidf_path = 'tuning_xgboost_tfidf_vectorizer.joblib'

    model, tfidf_vectorizer = load_model_and_tfidf(model_path, tfidf_path)

    user_input = input("Enter a review for sentiment prediction: ")

    prediction = predict([user_input], model, tfidf_vectorizer)

    print(f"Sentiment Prediction: {prediction[0]}")

if __name__ == "__main__":
    inference()