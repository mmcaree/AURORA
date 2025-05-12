# file: summarizer.py
import json
import asyncio
from transformers import pipeline

MEMORY_FILE = "aurora_diary.json"

# Load and format memory
def load_recent_memory(n=8):
    with open(MEMORY_FILE, 'r') as f:
        full = json.load(f)
        return full[-n:]

def format_for_summary(memory):
    return "\n".join(f"User: {m['user']}\nAurora: {m['ai']}" for m in memory)

# Load pipeline once
summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum", device=-1)

# Async summarization
async def summarize_recent_memory(n=8, max_tokens=64):
    memory = load_recent_memory(n)
    context = format_for_summary(memory)

    result = await asyncio.to_thread(
        summarizer,
        context,
        max_length=max_tokens,
        min_length=10,
        do_sample=False,
    )
    return result[0]['summary_text']
