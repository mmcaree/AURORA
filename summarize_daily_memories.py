import json
import chromadb
from sentence_transformers import SentenceTransformer
import os
from datetime import datetime

MEMORY_FILE = "chat_memory.json"
persist_directory = "E:/AURORA/chroma_memory"

# Init
client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_collection("aurora-memory")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Memory
with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
    memory = json.load(f)

# Summarize
summaries = []
current_day = datetime.now().strftime("%Y-%m-%d")

for entry in memory:
    summary = f"On {current_day}, user said: {entry['user']}, Aurora replied: {entry['ai']}."
    summaries.append(summary)

# Save into Chroma
for summary in summaries:
    embedding = embed_model.encode(summary).tolist()
    doc_id = str(abs(hash(summary)))

    collection.add(
        documents=[summary],
        metadatas=[{"tag": "daily-summary"}],
        embeddings=[embedding],
        ids=[doc_id]
    )

print("✅ Daily memory summaries imported into ChromaDB.")
