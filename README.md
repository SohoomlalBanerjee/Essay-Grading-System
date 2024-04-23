# Essay Grading System

## Summary
This project, titled "Essay Grading with LSTM Model," is a machine learning application that predicts the grade of an essay using a Long Short-Term Memory (LSTM) model trained on Word2Vec embeddings. Users input an essay, which undergoes preprocessing to remove stopwords, lemmatize words, and convert them to vectors. These vectors are then fed into the LSTM model, which predicts the essay grade.

## Demo

[![Demo Video](https://img.youtube.com/vi/-I5Yo4-MNGI/0.jpg)](https://youtu.be/-I5Yo4-MNGI)

### Streamlit Demo

Check out the live demo of the Essay Grading System using Streamlit:

[Essay Grading System](https://essay-grading-system.streamlit.app/)


## Credits
This project was developed by Archisman Ray, Soumedhik Bharati, and Sohoom Lal Banerjee for the SIT Hackathon. It was created as part of an effort to automate the grading process of essays using machine learning techniques.

## Model Links
The Google Drive for the two models are below. (>230 MB combined)
Model 1 - https://drive.google.com/file/d/1ihOKoAG2R1AJOXjW37ZRyU8n7nxbEJsB/view?usp=drive_link
Model 2 - https://drive.google.com/file/d/1mTazjmGrvgMqb5lsZegUJwj45U8DlpWC/view?usp=drive_link

## Installation
To run the application locally, follow these steps:
1. Clone the repository:
   ```
   git clone <repository_url>
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the pre-trained models from the provided links and place them in the project directory.

4. Run the application:
   ```
   streamlit run app.py
   ```
## Pre-trained Models
Download the pre-trained LSTM models from the following links:
- [Model 1](https://drive.google.com/file/d/1ihOKoAG2R1AJOXjW37ZRyU8n7nxbEJsB/view?usp=sharing)
- ![model1_diagram](https://github.com/Soumedhik/Essay-Grading-System/assets/113777577/0dfaf6ef-5de8-4ba8-85f7-bad1a0eab426)

- [Model 2](https://drive.google.com/file/d/1mTazjmGrvgMqb5lsZegUJwj45U8DlpWC/view?usp=sharing)
- ![model2_diagram](https://github.com/Soumedhik/Essay-Grading-System/assets/113777577/38777794-27f8-4602-a39b-8c0bfdbae500)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
