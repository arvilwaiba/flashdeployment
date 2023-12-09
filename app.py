# Import required libraries
import os
from flask import Flask, request, jsonify, render_template, Response
from flask_ngrok import run_with_ngrok
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import json

# Download stopwords and punkt resource
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords', download_dir='/opt/render/nltk_data')
    nltk.download('punkt', download_dir='/opt/render/nltk_data')

app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when the app is run

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
        text = data.get('text', '')  # Use an empty string as the default value if 'text' is not present

        if text is not None:
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
            predicted_sentiment = sentiment_mapping.get(predicted_label, 'unknown')

            # Explicitly set content type to JSON
            response = Response(response=json.dumps({'sentiment': predicted_sentiment}),
                                status=200,
                                mimetype="application/json")

            return response

        else:
            return Response(response=json.dumps({'error': 'Text is missing'}),
                            status=400,
                            mimetype="application/json")

    except Exception as e:
        app.logger.error(str(e))
        return Response(response=json.dumps({'error': 'Internal Server Error'}),
                        status=500,
                        mimetype="application/json")

if __name__ == '__main__':
    app.run(debug=True)
