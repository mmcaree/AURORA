import json
from transformers import pipeline

MEMORY_FILE = "chat_memory.json"

def load_recent_memory(n=8):
    with open(MEMORY_FILE, 'r') as f:
        full = json.load(f)
        return full[-n:]

def format_for_summary(memory):
    return "\n".join(f"User: {m['user']}\nAurora: {m['ai']}" for m in memory)

summarizer = pipeline("summarization", model="google/flan-t5-large")

if __name__ == "__main__":
    memory = load_recent_memory()
    raw_context = format_for_summary(memory)

    print(f"\n[INPUT TEXT]\n{raw_context}\n")

    result = summarizer(
        raw_context,
        max_length=64,
        min_length=10,
        do_sample=False
    )

    summary = result[0]['summary_text']
    print(f"\n[SUMMARY]\n{summary}")

