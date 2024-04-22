import streamlit as st
import numpy as np
import re
from gensim.models import KeyedVectors
from keras.models import Sequential, load_model, Model
from keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from keras.layers import Attention
from keras.regularizers import l2
import tensorflow as tf
import google.generativeai as genai
import pathlib
import textwrap
import pytesseract
import shutil
import os
import random
try:
 from PIL import Image
except ImportError:
 import Image
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

api_key = st.secrets["GEMINI_API_KEY"] 
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')

def get_model2():
    model = Sequential()
    model.add(BatchNormalization(input_shape=[1, 600]))
    model.add(Bidirectional(LSTM(512, dropout=0.4, recurrent_dropout=0.3, return_sequences=True, kernel_regularizer=l2(0.001))))
    model.add(Bidirectional(LSTM(256, dropout=0.4, recurrent_dropout=0.3, return_sequences=True, kernel_regularizer=l2(0.001))))
    model.add(Bidirectional(LSTM(128, dropout=0.4, recurrent_dropout=0.3, kernel_regularizer=l2(0.001))))
    model.add(Dropout(0.6))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.6))
    model.add(Dense(11, activation='softmax'))
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

lstm_model2 = get_model2()
lstm_model2.load_weights("Streamlit_Model/essay_rank_lstm_2.keras")

w2v_path = 'word2vecmodel.bin'
word2vec_model = KeyedVectors.load(w2v_path, mmap='r')

def essay_to_vector(essay, model):
    stop_words = set(stopwords.words('english'))
    essay = re.sub("[^A-Za-z]", " ", essay).lower()
    words = word_tokenize(essay)
    words = [w for w in words if w not in stop_words]
    essay_vec = np.zeros((model.vector_size,), dtype="float32")
    no_of_words = 0
    for word in words:
        if word in model:
            no_of_words += 1
            essay_vec = np.add(essay_vec, model[word])
    if no_of_words != 0:
        essay_vec = np.divide(essay_vec, no_of_words)
    return essay_vec

def reshape_for_lstm(vector):
    return np.reshape(vector, (1, 1, -1))

def predict_score(essay):
    vector = essay_to_vector(essay, word2vec_model)
    vector = reshape_for_lstm(vector)
    prediction = lstm_model2.predict(vector)
    score = np.argmax(prediction)
    if (score<6)
     bias = 2
    else
     bias = 1 
    adjusted_score = score + bias
    adjusted_score = min(adjusted_score, 10)
    return adjusted_score

def main():
    st.set_page_config(page_title="Essay Score Predictor", page_icon="📝")

    st.markdown("""
        <style>
        .title {
            color: #ff6347;
            animation: color-change 2s infinite;
            text-align: center;
        }
        @keyframes color-change {
            0% { color: #ff6347; }
            50% { color: #4682b4; }
            100% { color: #ff6347; }
        }
        </style>
        <h1 class="title">SIT Hackathon '24 Project</h1>
        """, unsafe_allow_html=True)

    st.markdown("""
        <h2 style='text-align: center; color: #2e8b57;'>
        "Unveiling the Art of Automated Essay Grading: AI's Journey to Explainability"
        </h2>
        """, unsafe_allow_html=True)

    st.markdown("""
        <style>
        .subheader-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .subheader {
            opacity: 0;
            animation: fade-in 1s forwards;
        }
        @keyframes fade-in {
            100% { opacity: 1; }
        }
        </style>
        <div class="subheader-container">
            <h3 class="subheader">Develop an AI model that not only grades essays but also elucidates the score.</h3>
            <h3 class="subheader">-By Sohoom Lal Banerjee, Soumedhik Bharati and Archisman Ray.</h3>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## Enter your essay below to predict its score.")

    user_essay = st.text_area("Paste your essay here:", height=300)

    if st.button("Predict Score"):
        predicted_score = predict_score(user_essay)
        st.success(f"The predicted score for the essay is: {predicted_score} out of 10.")
        prompt = f"Justify rating the essay '{user_essay}' as {predicted_score} out of 10 and discuss its highs and lows and the justification behind marking it as such. Also suggest improvements in the end which could possibly address the issues with the essay."
        response = model.generate_content(prompt)
        analysis_text = response.text

        st.markdown(f"""
            <div class="explanation">
            <h4>Explanation:</h4>
            {analysis_text}
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
