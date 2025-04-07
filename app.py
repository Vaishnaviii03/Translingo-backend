from flask import Flask, request, jsonify
from keras.models import load_model
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
from flask_cors import CORS
import os

app = Flask(__name__)

# Allow requests from localhost (for development) and your Netlify frontend (for production)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://dashing-starship-f27d7a.netlify.app"]}})

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
        # Convert English sentence to sequences
        seq = english_tokenizer.texts_to_sequences([english_sentence])
        padded_seq = pad_sequences(seq, maxlen=max_french_sequence_length, padding='post')
        
        # Predict the French sentence
        pred = model.predict(padded_seq)
        pred = np.argmax(pred, axis=-1)
        
        # Convert predicted sequence to French sentence
        french_sentence = ' '.join(french_tokenizer.index_word.get(i, '<UNK>') for i in pred[0] if i != 0)
        return french_sentence
    except Exception as e:
        print(f"Error during translation: {e}")
        return "Translation Error"

# API endpoint for React to call
@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()  # Get the JSON data from the request
    english_sentence = data.get('sentence', '')  # Extract the sentence to translate
    
    if english_sentence:
        # Call the translation function
        french_translation = translate_sentence(english_sentence)
        return jsonify({'translation': french_translation})  # Send the translation as JSON
    else:
        return jsonify({'translation': 'No sentence provided'})  # Handle empty sentence input

# Default home route
@app.route('/')
def home():
    return "Welcome to TransLingo API! Use /translate endpoint with POST request."

if __name__ == "__main__":
    # Set the port to use for the backend
    port = int(os.environ.get("PORT", 5000))  # Use PORT environment variable or default to 5000
    app.run(host="0.0.0.0", port=port, debug=True)  # Start Flask app

