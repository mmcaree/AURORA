from pathlib import Path
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_FILE = "faiss_index.index"
MEMORY_FILE = "faiss_memory.json"
DIM = 384  # Embedding size for all-MiniLM-L6-v2

model = SentenceTransformer("all-MiniLM-L6-v2")

class FaissMemoryManager:
    def __init__(self, index_path=INDEX_FILE, memory_path=MEMORY_FILE):
        self.index_path = Path(index_path)
        self.memory_path = Path(memory_path)
        self.index = None
        self.memory = []
        self.load()

    def load(self):
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = faiss.IndexFlatL2(DIM)

        if self.memory_path.exists():
            with open(self.memory_path, "r", encoding="utf-8") as f:
                self.memory = json.load(f)

    def save(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, indent=2)

    def add_entry(self, text):
        embedding = model.encode([text])
        self.index.add(np.array(embedding).astype("float32"))
        self.memory.append(text)
        self.save()

    def search(self, query, top_k=3):
        if not self.memory:
            return []
        embedding = model.encode([query])
        D, I = self.index.search(np.array(embedding).astype("float32"), top_k)
        return [self.memory[i] for i in I[0] if i < len(self.memory)]
