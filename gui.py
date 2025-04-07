import tkinter as tk
from tkinter import messagebox, font
from tensorflow.keras.models import load_model
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np

# Load the model
model = load_model('english_to_french_model.keras')

# Load the tokenizers
with open('english_tokenizer.json', 'r', encoding='utf8') as f:
    english_tokenizer = tokenizer_from_json(json.load(f))
with open('french_tokenizer.json', 'r', encoding='utf8') as f:
    french_tokenizer = tokenizer_from_json(json.load(f))

# Load max sequence length
with open('sequence_length.json', 'r', encoding='utf8') as f:
    max_french_sequence_length = json.load(f)

def translate_sentence(english_sentence):
    try:
        # Tokenize and pad the input sentence
        seq = english_tokenizer.texts_to_sequences([english_sentence])
        padded_seq = pad_sequences(seq, maxlen=max_french_sequence_length, padding='post')
        
        # Predict the French sentence
        pred = model.predict(padded_seq)
        pred = np.argmax(pred, axis=-1)
        
        # Convert token IDs to words
        french_sentence = ' '.join(french_tokenizer.index_word.get(i, '<UNK>') for i in pred[0] if i != 0)
        
        return french_sentence
    except Exception as e:
        print(f"Error during translation: {e}")
        return "Translation Error"

def on_translate_button_click():
    english_sentence = entry.get()
    if english_sentence:
        french_sentence = translate_sentence(english_sentence)
        result_var.set(french_sentence)
    else:
        messagebox.showwarning("Input Error", "Please enter a sentence.")

# GUI setup
root = tk.Tk()
root.title("English to French Translator")

# Set window size and center it
root.geometry("500x300")
root.resizable(False, False)

# Create a custom font
custom_font = font.Font(family="Helvetica", size=12)

# Add padding and styling
padding = 10

tk.Label(root, text="Enter English Sentence:", font=custom_font).pack(pady=padding, padx=padding, anchor='w')
entry = tk.Entry(root, width=60, font=custom_font)
entry.pack(pady=padding, padx=padding, anchor='w')

tk.Button(root, text="Translate", command=on_translate_button_click, font=custom_font).pack(pady=padding)

result_var = tk.StringVar()
tk.Label(root, text="French Translation:", font=custom_font).pack(pady=padding, padx=padding, anchor='w')
result_label = tk.Label(root, textvariable=result_var, font=custom_font, wraplength=450, anchor='w', justify='left')
result_label.pack(pady=padding, padx=padding, anchor='w')

root.mainloop()
