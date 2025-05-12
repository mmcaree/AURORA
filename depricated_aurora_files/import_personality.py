# file: import_personality_csv.py

import csv
import chromadb
from sentence_transformers import SentenceTransformer
import os

CSV_FILE = "E:/AURORA/personality.csv"  # <-- path to your CSV
persist_directory = "E:/AURORA/chroma_memory"

# Init
client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_collection("aurora-memory")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Read CSV
with open(CSV_FILE, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        user_input = row['input']
        ai_response = row['output']
        
        combined = f"User: {user_input}\nAurora: {ai_response}"
        embedding = embed_model.encode(combined).tolist()

        doc_id = str(abs(hash(combined)))  # Stable ID
        collection.add(
            documents=[combined],
            metadatas=[{"tag": "personality"}],
            embeddings=[embedding],
            ids=[doc_id]
        )

print("✅ Personality CSV imported into ChromaDB.")
