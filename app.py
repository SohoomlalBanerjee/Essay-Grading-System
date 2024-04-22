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

api_key = 'GEMINI_API_KEY'  
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')

def get_model1():
    input_layer = Input(shape=[1, 600])
    bn_input = BatchNormalization()(input_layer)
    lstm1 = Bidirectional(LSTM(1024, dropout=0.3, recurrent_dropout=0.3, return_sequences=True, kernel_regularizer=l2(0.001)))(bn_input)
    lstm2 = Bidirectional(LSTM(512, dropout=0.3, recurrent_dropout=0.3, return_sequences=True, kernel_regularizer=l2(0.001)))(lstm1)
    lstm3 = Bidirectional(LSTM(256, dropout=0.3, recurrent_dropout=0.3, return_sequences=True, kernel_regularizer=l2(0.001)))(lstm2)
    attention = Attention()([lstm3, lstm3])
    dropout = Dropout(0.6)(attention)
    dense1 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(dropout)
    dropout2 = Dropout(0.6)(dense1)
    output_layer = Dense(11, activation='softmax')(dropout2)
    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

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

lstm_model1 = get_model1()
lstm_model1.load_weights("essay_rank_lstm_1.keras")

lstm_model2 = get_model2()
lstm_model2.load_weights("essay_rank_lstm_2.keras")

w2v_path = 'word2vecmodel.bin'
word2vec_model = KeyedVectors.load(w2v_path, mmap='r')

def perform_ocr(image):
    extracted_text = pytesseract.image_to_string(image)
    return extracted_text

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
    bias = 2
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

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    user_essay = st.text_area("Essay:", height=300)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        extracted_text = perform_ocr(image)
    else:
        extracted_text = user_essay

    st.markdown("""
        <style>
        .stButton>button:hover {
            color: #ffffff;
            background-color: #ff6347;
            transform: scale(1.05);
        }
        </style>
        """, unsafe_allow_html=True)
    
    if st.button("Predict Score"):
        predicted_score = predict_score(extracted_text)
        st.success(f"The predicted score for the essay is: {predicted_score} out of 10.")
        prompt = f"Justify rating the essay '{extracted_text}' as {predicted_score} out of 10 and discuss its highs and lows and the justification behind marking it as such."
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
