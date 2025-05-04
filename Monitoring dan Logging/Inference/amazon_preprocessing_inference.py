import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import string

nltk.download('punkt')
nltk.download('stopwords')

def preprocessing_text(text):
    if not isinstance(text, str):
        text = str(text)

    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'https?://\S+|www.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip().lower()

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]

    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in tokens])

    return text

def preprocess_inference_data(text_data):
    processed_data = text_data.apply(preprocessing_text)
    return processed_data

def transform_text_to_tfidf(text_data, tfidf_vectorizer):
    text_tfidf = tfidf_vectorizer.transform(text_data)
    return text_tfidf
