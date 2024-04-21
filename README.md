### Essay Grading System

This project is a machine learning application that predicts the score of an essay based on its content. It utilizes a Long Short-Term Memory (LSTM) neural network along with Word2Vec embeddings for text processing.

### Overview

The Essay Grading System is designed to automatically evaluate the quality of essays by predicting their scores. It leverages natural language processing techniques and deep learning to analyze the content of essays and assign them scores ranging from very poor to phenomenal.

### Features

- **LSTM Model:** The system uses a Long Short-Term Memory (LSTM) neural network, a type of recurrent neural network (RNN), to analyze the textual content of essays and predict their scores.
  
- **Word2Vec Embeddings:** Word2Vec embeddings are used to represent words in a continuous vector space, capturing semantic meanings of words and enhancing the model's understanding of the essay's content.
  
- **Pre-trained Models:** Pre-trained LSTM models are provided to ensure accurate and efficient scoring of essays.

### Usage

To use the Essay Grading System:

1. Provide the essay text.
2. Click on the "Predict Score" button.
3. The system will analyze the essay and display the predicted score.

### Pre-trained Models

You can download the pre-trained LSTM models from the following links:

- [Model 1](https://drive.google.com/file/d/1ihOKoAG2R1AJOXjW37ZRyU8n7nxbEJsB/view?usp=sharing)
- [Model 2](https://drive.google.com/file/d/1mTazjmGrvgMqb5lsZegUJwj45U8DlpWC/view?usp=sharing)

### Installation

To run the project locally, follow these steps:

1. Clone the repository:
   ```
   git clone <repository_url>
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the pre-trained models and place them in the project directory.

4. Run the application:
   ```
   streamlit run app.py
   ```

### Credits

This project was developed by [Your Name] and [Contributor Name] as part of [Project Name] at [Organization/University].

### License

This project is licensed under the [MIT License](LICENSE).
