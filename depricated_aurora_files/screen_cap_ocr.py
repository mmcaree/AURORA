# file: screen_capture_ocr.py

import pytesseract
import pygetwindow as gw
from PIL import ImageGrab
from datetime import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
import time

# Setup persistent ChromaDB
persist_directory = os.path.abspath("E:/AURORA/chroma_memory")
chroma_client = chromadb.PersistentClient(path=persist_directory)

if "aurora-memory" not in [c.name for c in chroma_client.list_collections()]:
    chroma_client.create_collection("aurora-memory")

vector_store = chroma_client.get_collection("aurora-memory")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Perform OCR on screen
def capture_screen_text():
    img = ImageGrab.grab()
    text = pytesseract.image_to_string(img)
    return text.strip()

def get_window_title():
    try:
        return gw.getActiveWindowTitle() or "Unknown"
    except:
        return "Unknown"

# Store OCR text into ChromaDB with tags
def store_screen_ocr():
    screen_text = capture_screen_text()
    if not screen_text.strip():
        print("[INFO] No screen text detected.")
        return

    window = get_window_title()
    timestamp = datetime.now().isoformat()
    tag = "ocr"
    content = f"Screen OCR from {window} at {timestamp}:
{screen_text}"

    embedding = embedder.encode(content).tolist()
    doc_id = str(int(time.time() * 1000))

    vector_store.add(
        documents=[content],
        metadatas=[{"window": window, "timestamp": timestamp, "tag": tag}],
        embeddings=[embedding],
        ids=[doc_id]
    )

    #print(f"[SAVED] Captured screen from: {window}\nText Length: {len(screen_text)}\n")

if __name__ == "__main__":
    store_screen_ocr()
