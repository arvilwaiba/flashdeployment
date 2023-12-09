import os
import re
import traceback
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

model = keras.models.load_model(model_file_path)

max_words = 10000
max_len = 100
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    preprocessed_text = ' '.join(words) if words else ''
    return preprocessed_text

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment_route():
    try:
        data = request.get_json(force=True)
        text = data.get('text', '')

        preprocessed_text = preprocess_text(text)

        if not preprocessed_text:
            print("Error: Empty preprocessed text")
            return jsonify({'error': 'Empty preprocessed text'}), 400

        # Tokenize and pad sequence
        sequence = tokenizer.texts_to_sequences([preprocessed_text])

        if not sequence:
            print("Error: Unable to convert text to sequence")
            return jsonify({'error': 'Unable to convert text to sequence'}), 400

        padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')

        print("Padded Sequence:", padded_sequence)

        prediction = sentiment_model.predict(padded_sequence)
        predicted_label = np.argmax(prediction)

        sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        predicted_sentiment = sentiment_mapping[predicted_label]

        response = Response(response=json.dumps({'sentiment': predicted_sentiment}),
                            status=200,
                            mimetype="application/json")

        return response

    except Exception as e:
        print("Exception:", e)
        traceback.print_exc()
        return jsonify({'error': 'Internal Server Error'}), 500


if __name__ == '__main__':
    app.run()
