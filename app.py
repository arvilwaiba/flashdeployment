import os
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
from flask_cors import CORS
from flask import Response, json
import traceback  # Import the traceback module

# Download stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Download punkt tokenizer
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = Flask(__name__)
run_with_ngrok(app)
CORS(app)

# Get the absolute path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the path to your model file (assuming it's in the same directory)
model_file_path = os.path.join(current_dir, 'model.h5')

# Load your pre-trained sentiment analysis model
model = keras.models.load_model(model_file_path)

# Load the Tokenizer
max_words = 10000
max_len = 100
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")

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
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment_route():
    try:
        data = request.get_json()
        text = data.get('text')  # Use get to handle the case where 'text' is not in the JSON

        # Print input text for debugging
        print("Input Text:", text)

        if text is None:
            return jsonify({'error': 'Missing or invalid input text'}), 400

        # Preprocess the text
        preprocessed_text = preprocess_text(text)

        # Print preprocessed text for debugging
        print("Preprocessed Text:", preprocessed_text)

        # Tokenize and pad the sequence
        sequence = tokenizer.texts_to_sequences([preprocessed_text])

        # Handle the case where the sequence is None or empty
        if not sequence or not sequence[0]:
            return jsonify({'error': 'Unable to tokenize the input text'}), 400

        padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')

        # Make prediction
        prediction = model.predict(padded_sequence)
        predicted_label = np.argmax(prediction)

        # Map predicted label to sentiment class
        sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        predicted_sentiment = sentiment_mapping[predicted_label]

        # Explicitly set content type to JSON
        response = Response(response=json.dumps({'sentiment': predicted_sentiment}),
                            status=200,
                            mimetype="application/json")

        return response

    except Exception as e:
        print("Exception:", e)
        traceback.print_exc()  # Print traceback for detailed error information
        return jsonify({'error': 'Internal Server Error'}), 500

