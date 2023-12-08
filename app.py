
# Install necessary libraries
!pip install flask
!pip install flask-ngrok
!pip install flask-cors

# Import required libraries
from flask import Flask, request, jsonify, render_template
from flask_ngrok import run_with_ngrok
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk


# Download stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)
CORS(app)
run_with_ngrok(app)  # Start ngrok when the app is run

# Specify the path to your model file
model_file_path = '/Users/arvilwaiba/desktop/flash/model.h5'

# Load your pre-trained sentiment analysis model
model = keras.models.load_model(model_file_path)

# Load the Tokenizer
max_words = 10000
max_len = 100
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts([])  # You might want to use your training data here

# Remove stopwords
stop_words = set(stopwords.words('english'))

# Preprocess text function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Define Flask routes
@app.route('https://sentimentanalysis-arvils-projects.vercel.app/')
def index():
    return render_template('index.html')

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment_route():
    data = request.get_json()
    text = data['text']

    # Preprocess the text
    preprocessed_text = preprocess_text(text)

    # Tokenize and pad the sequence
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')

    # Make prediction
    prediction = model.predict(padded_sequence)
    predicted_label = np.argmax(prediction)

    # Map predicted label to sentiment class
    sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    predicted_sentiment = sentiment_mapping[predicted_label]

    return jsonify({'sentiment': predicted_sentiment})
