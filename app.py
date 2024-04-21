import streamlit as st
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from keras.models import Sequential, load_model, Model
from keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, TimeDistributed, Attention, BatchNormalization
from keras.regularizers import l2
import tensorflow as tf
import pathlib
import textwrap
import google.generativeai as genai
from google.colab import userdata
from IPython.display import display, Markdown
from google.colab import userdata

api_key = userdata.get('GEMINI_API_KEY')
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

w2v_path = '/kaggle/working/word2vecmodel.bin'
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
    prediction = (lstm_model1.predict(vector)+lstm_model2.predict(vector))/2
    score = np.argmax(prediction)
    return score

def main():
    st.title("Essay Score Predictor")
    st.write("Enter your essay below to predict its score.")

    user_essay = st.text_area("Essay")
    
    if st.button("Predict Score"):
        predicted_score = predict_score(user_essay)
        st.write(f"The predicted score for the essay is: {predicted_score}")
        prompt = f"Justify rating the essay {user_essay} as {predicted_score} out of 10 and discuss its highs and lows and the justification behind marking it as such."
        response = model.generate_content(prompt)
        analysis_text = response.text
        st.write(f"Explanation:"{analysis_text})
if __name__ == "__main__":
    main()
