import os
import re
import traceback
import logging
from flask import Flask, request, jsonify, Response, json
from flask_cors import CORS
from flask_ngrok import run_with_ngrok
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import nltk

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
CORS(app)
run_with_ngrok(app)  # Start ngrok when the app is run

current_dir = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(current_dir, 'model.h5')

try:
    model = keras.models.load_model(model_file_path)
except keras.utils.HDF5MatrixError as e:
    logging.error(f"Error loading the model: {e}")
    raise SystemExit(1)  # Exit the application if model loading fails

max_words = 10000
max_len = 100
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    logging.info("Original Text: %s", text)
    
    # Check for empty input
    if not text:
        logging.warning("Empty input text")
        return None, None
    
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    
    # Check if there are valid words after removing stopwords
    if not words:
        logging.warning("No valid words after removing stopwords")
        return None, None
    
    preprocessed_text = ' '.join(words)
    logging.info("Preprocessed Text: %s", preprocessed_text)
    return preprocessed_text, words

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment_route():
    try:
        data = request.get_json(force=True)
        text = data.get('text', '')

        preprocessed_text, words = preprocess_text(text)

        if preprocessed_text is None:
            return jsonify({'error': 'Empty or invalid input text'}), 400

        logging.info("Preprocessed Text: %s", preprocessed_text)

        if not words:
            return jsonify({'error': 'No valid words after removing stopwords'}), 400

        sequence = tokenizer.texts_to_sequences([preprocessed_text])

        if not sequence or not sequence[0]:
            return jsonify({'error': 'Unable to tokenize the input text'}), 400

        logging.info("Tokenized Sequence: %s", sequence)

        padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
        logging.info("Padded Sequence: %s", padded_sequence)

        prediction = model.predict(padded_sequence)
        predicted_label = np.argmax(prediction)

        sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        predicted_sentiment = sentiment_mapping[predicted_label]

        return jsonify({'sentiment': predicted_sentiment})

    except Exception as e:
        logging.error("Exception: %s", e)
        traceback.print_exc()
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run()
