from flask import Flask, request, jsonify
from keras.models import load_model
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
from flask_cors import CORS
import os

app = Flask(__name__)

# Allow all origins temporarily — restrict later in production
CORS(app)

# Add CORS headers manually after every request (helps with OPTIONS preflight)
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# Load model and tokenizers
model = load_model('english_to_french_model.keras')

with open('english_tokenizer.json', 'r', encoding='utf8') as f:
    english_tokenizer = tokenizer_from_json(json.load(f))

with open('french_tokenizer.json', 'r', encoding='utf8') as f:
    french_tokenizer = tokenizer_from_json(json.load(f))

with open('sequence_length.json', 'r', encoding='utf8') as f:
    max_french_sequence_length = json.load(f)

# Translation function
def translate_sentence(english_sentence):
    try:
        seq = english_tokenizer.texts_to_sequences([english_sentence])
        padded_seq = pad_sequences(seq, maxlen=max_french_sequence_length, padding='post')
        pred = model.predict(padded_seq)
        pred = np.argmax(pred, axis=-1)
        french_sentence = ' '.join(french_tokenizer.index_word.get(i, '<UNK>') for i in pred[0] if i != 0)
        return french_sentence
    except Exception as e:
        print(f"Error during translation: {e}")
        return "Translation Error"

# API endpoint for translation
@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    english_sentence = data.get('sentence', '')
    if english_sentence:
        french_translation = translate_sentence(english_sentence)
        return jsonify({'translation': french_translation})
    else:
        return jsonify({'translation': 'No sentence provided'})

# Home route
@app.route('/')
def home():
    return "Welcome to TransLingo API! Use /translate endpoint with POST request."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
