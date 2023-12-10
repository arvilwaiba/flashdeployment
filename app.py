import os
import re
import traceback
from flask import Flask, request, jsonify, Response, json
from flask_cors import CORS
from flask_ngrok import run_with_ngrok
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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
tokenizer = None  # Tokenizer will be created dynamically during app initialization
model = None

max_words = 10000
max_len = 100
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize using nltk.word_tokenize
    words = word_tokenize(text)
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def initialize_models():
    global model, tokenizer
    # Load the sentiment analysis model
    model = keras.models.load_model(model_file_path)

    # Dummy training data (replace this with your actual training data)
    train_data = ["This is a positive sentence.", "This is a negative sentence.", "This is a neutral sentence."]
    
    # Create and fit the tokenizer
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_data)

# Initialize models when the app starts
initialize_models()

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
        padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')

        print("Padded Sequence:", padded_sequence)

        prediction = model.predict(padded_sequence)
        predicted_label = np.argmax(prediction)

        sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        predicted_sentiment = sentiment_mapping[predicted_label]

        print("Predicted Sentiment:", predicted_sentiment)

        response = Response(response=json.dumps({'sentiment': predicted_sentiment}),
                            status=200,
                            mimetype="application/json")

        print("Entire Response:", response.response)

        return response

    except Exception as e:
        print("Exception:", e)
        traceback.print_exc()
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run()
