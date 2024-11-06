from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import os

app = Flask(__name__)

# Laden Sie Ihr Chat-Modell
model_name = "dbmdz/german-gpt2"  # Ersetzen Sie dies durch Ihr Modell
tokenizer = AutoTokenizer.from_pretrained(model_name)
chat_model = AutoModelForCausalLM.from_pretrained(model_name)

# Prüfen und setzen des pad_token, falls notwendig
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Sammlungen für LocalDocs
collections = []

def load_collections():
    global collections
    # Hier können Sie Ihre Sammlungen laden oder initialisieren
    # Beispiel: Laden von Sammlungen aus einem Ordner
    collections = []
    collections_folder = 'collections'
    if not os.path.exists(collections_folder):
        os.makedirs(collections_folder)
    for collection_name in os.listdir(collections_folder):
        collection_path = os.path.join(collections_folder, collection_name)
        if os.path.isdir(collection_path):
            # Laden Sie die Sammlung
            collection = load_collection(collection_path)
            collections.append(collection)

def load_collection(collection_path):
    # Implementieren Sie das Laden Ihrer Sammlung
    # Dies ist ein Platzhalter
    collection = {
        "name": os.path.basename(collection_path),
        "folder": collection_path,
        "filenames": [],
        "texts": [],
        "index": None,
        "model": SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    }
    return collection

load_collections()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '')
    use_localdocs = data.get('use_localdocs', False)
    conversation_history = data.get('conversation_history', '')

    if use_localdocs:
        # Kontext aus LocalDocs abrufen
        localdocs_context = get_localdocs_context(user_input)
        if localdocs_context:
            conversation_history += f"{localdocs_context}\n"

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

def get_localdocs_context(query):
    # Überprüfen, ob Sammlungen vorhanden sind
    if not collections:
        return ""

    # Kontext sammeln
    context = ""
    for collection in collections:
        index = collection['index']
        model = collection['model']
        texts = collection['texts']

        # Anfrage einbetten
        query_embedding = model.encode([query], convert_to_numpy=True)

        # Index durchsuchen
        k = 3  # Anzahl der abzurufenden relevanten Dokumente
        distances, indices = index.search(query_embedding, k)

        # Relevante Texte sammeln
        for idx in indices[0]:
            if idx < len(texts):
                context += texts[idx] + "\n\n"

    # Kontext begrenzen, um zu lange Eingaben zu vermeiden
    max_context_length = 1000  # Maximal 1000 Zeichen im Kontext
    if len(context) > max_context_length:
        context = context[:max_context_length] + "..."

    return context.strip()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
