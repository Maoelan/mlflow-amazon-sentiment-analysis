import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import string
import os

nltk.download('vader_lexicon')
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

def assign_satisfaction_label(text):
    cleaned_text = preprocessing_text(text)

    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(cleaned_text)
    compound_score = sentiment_score['compound']

    if compound_score >= 0.05:
        return 'Satisfied'
    elif compound_score <= -0.05:
        return 'Dissatisfied'
    else:
        return 'Neutral'

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data_raw = data[['content']]
    
    sentiment_data = data_raw.copy()

    sentiment_data['content'] = sentiment_data['content'].apply(preprocessing_text)
    sentiment_data.dropna(inplace=True)
    sentiment_data.drop_duplicates(inplace=True)

    sentiment_data['satisfaction_label'] = sentiment_data['content'].apply(assign_satisfaction_label)

    label_encoder = LabelEncoder()
    sentiment_data['satisfaction_label'] = label_encoder.fit_transform(sentiment_data['satisfaction_label'])

    return sentiment_data