# # Import necessary libraries
# import pandas as pd
# import numpy as np
# import re
# import string
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# from nltk.corpus import stopwords
# import nltk
#
# # Download stopwords if not already downloaded
# nltk.download('stopwords')
#
# # Load your dataset
# df = pd.read_csv(r"C:\Users\Vivek\PycharmProjects\cyber\aggression_parsed_dataset.csv")  # Fixed file path
# # Check the shape of the dataset
# print("Dataset Shape: ", df.shape)
#
# # 1. Data Preprocessing
#
# # Define a function to clean text data
# def clean_text(text):
#     # Convert to lowercase
#     text = text.lower()
#     # Remove URLs
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
#     # Remove punctuation
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     # Remove numbers
#     text = re.sub(r'\d+', '', text)
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     text = " ".join([word for word in text.split() if word not in stop_words])
#     return text
#
# # Apply the cleaning function to the Text column
# df['cleaned_text'] = df['Text'].apply(clean_text)
#
# # Check the first few rows to verify cleaning
# print(df.head())
#
# # 2. Feature Extraction (TF-IDF Vectorization)
#
# # Define the TF-IDF Vectorizer
# tfidf = TfidfVectorizer(max_features=5000)
#
# # Fit and transform the cleaned text
# X = tfidf.fit_transform(df['cleaned_text']).toarray()
#
# # Labels (Target Variable)
# y = df['oh_label']
#
# # 3. Train-Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# # 4. Model Building (Logistic Regression)
#
# # Initialize the model
# model = LogisticRegression()
#
# # Train the model
# model.fit(X_train, y_train)
#
# # 5. Model Evaluation
#
# # Make predictions
# y_pred = model.predict(X_test)
#
# # Evaluate the model
# print("Accuracy: ", accuracy_score(y_test, y_pred))
# print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
# print("Classification Report: \n", classification_report(y_test, y_pred))
#
# # Save the model if you want to deploy it later
# import joblib
# joblib.dump(model, 'cyberbullying_detection_model.pkl')
# joblib.dump(tfidf, 'tfidf_vectorizer.pkl')


import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load the saved model and TF-IDF vectorizer
model = joblib.load('cyberbullying_detection_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Function to clean text (same as the one you used before)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Streamlit app
st.title("Cyberbullying Detection System")

# Create a text input box for user input
comment = st.text_area("Enter a comment to check if it is cyberbullying")

# Button to trigger prediction
if st.button("Detect"):
    if comment:
        # Clean the input text
        cleaned_comment = clean_text(comment)

        # Transform the cleaned text with the TF-IDF vectorizer
        vectorized_comment = tfidf.transform([cleaned_comment]).toarray()

        # Make prediction using the loaded model
        prediction = model.predict(vectorized_comment)

        # Map the prediction to a readable result
        if prediction == 0:
            st.success("This comment is Non-Cyberbullying.")
        else:
            st.error("This comment is Cyberbullying.")
    else:
        st.warning("Please enter a comment to analyze.")

# Optional footer or description
st.write("This application detects whether a comment is cyberbullying or not using a trained machine learning model.")
