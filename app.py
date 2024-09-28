import streamlit as st
import pandas as pd
import numpy as np
import joblib
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load the trained model
model = joblib.load('log_reg_model.pkl')

# Initialize NLP tools
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

# Function to make prediction
def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    sentiment_score = TextBlob(preprocessed_text).sentiment.polarity
    sentiment = 'Neutral'  # Default sentiment
    
    if sentiment_score > 0:
        sentiment = 'Positive'
    elif sentiment_score < 0:
        sentiment = 'Negative'
    
    return sentiment

# Create Streamlit app
def main():
    st.title("Sentiment Analysis App")

    # User input for review text
    review_text = st.text_area("Enter your review:", "")

    # Make prediction
    if st.button('Predict Sentiment'):
        if review_text:
            sentiment = predict_sentiment(review_text)
            st.write(f"Predicted Sentiment: {sentiment}")
        else:
            st.write("Please enter a review text.")

# Run the app
if __name__ == '__main__':
    main()
