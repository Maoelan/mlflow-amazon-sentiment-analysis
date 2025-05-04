import joblib
from amazon_preprocessing_inference import preprocessing_text, transform_text_to_tfidf

# Fungsi untuk memuat model dan TFIDF Vectorizer
def load_model_and_tfidf(model_path, tfidf_path):
    model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(tfidf_path)
    return model, tfidf_vectorizer

# Fungsi untuk melakukan prediksi
def predict(text_data, model, tfidf_vectorizer):
    # Proses data teks tanpa menggunakan apply() karena text_data adalah list
    processed_text = [preprocessing_text(text) for text in text_data]

    # Transformasi teks ke dalam bentuk TF-IDF
    text_tfidf = transform_text_to_tfidf(processed_text, tfidf_vectorizer)

    # Melakukan prediksi
    predictions = model.predict(text_tfidf)
    
    # Menerjemahkan prediksi numerik ke dalam label kategorikal
    sentiment_labels = {0: "Dissatisfied", 1: "Neutral", 2: "Satisfied"}
    categorical_prediction = [sentiment_labels.get(pred, "Unknown") for pred in predictions]
    
    return categorical_prediction

# Fungsi untuk inference sekali saja
def single_inference():
    model_path = 'tuning_xgboost_best_model.pkl'
    tfidf_path = 'tuning_xgboost_tfidf_vectorizer.pkl'
    
    # Memuat model dan tfidf vectorizer
    model, tfidf_vectorizer = load_model_and_tfidf(model_path, tfidf_path)
    
    # Menerima input dari pengguna
    user_input = input("Enter a review for sentiment prediction: ")

    # Melakukan prediksi
    prediction = predict([user_input], model, tfidf_vectorizer)

    # Menampilkan hasil prediksi
    print(f"Sentiment Prediction: {prediction[0]}")

# Menjalankan aplikasi tanpa looping
if __name__ == "__main__":
    single_inference()
