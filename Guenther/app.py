from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import os
import numpy as np

app = Flask(__name__)

# Laden Sie Ihr Chat-Modell
model_name = "dbmdz/german-gpt2"  # Ersetzen Sie dies durch Ihr gewünschtes Modell
tokenizer = AutoTokenizer.from_pretrained(model_name)
chat_model = AutoModelForCausalLM.from_pretrained(model_name)

# Prüfen und setzen des pad_token, falls notwendig
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form.get('message', '')
    conversation_history = request.form.get('conversation_history', '')

    # Eingabe vorbereiten
    inputs = tokenizer(
        conversation_history + f"Sie: {user_input}\n",
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=1024
    )

    # Antwort generieren
    outputs = chat_model.generate(
        **inputs,
        max_length=150,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        repetition_penalty=1.2,
    )

    # Ausgabe dekodieren
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Aktualisieren der Gesprächshistorie
    conversation_history += f"Sie: {user_input}\nKI: {response}\n"

    return jsonify({'response': response, 'conversation_history': conversation_history})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
