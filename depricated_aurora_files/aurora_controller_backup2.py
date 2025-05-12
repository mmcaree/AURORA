# file: aurora_controller.py

import subprocess
import time
import os
import sys
from datetime import datetime
from typing import List, Dict
import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import threading
from textblob import TextBlob
from RealtimeTTS import TextToAudioStream, KokoroEngine
import keyboard
import re
import queue
from functools import lru_cache
from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
import asyncio
import websockets
import numpy as np
from summarizer import summarize_recent_memory
from sklearn.cluster import KMeans
from aurora_listen import (
    start_pruning_loop,
    wait_for_hotkey,
    detect_memory_tag,
    clean_ai_response,
    is_terminal_focused
)
from asyncio import Lock

voice_mode = False
input_queue = queue.Queue()

_summary_lock = Lock()
_summary_cache = {"length": 0, "text": ""}
_summary_last_update = 0
_summary_refresh_interval = 300
_summary_future = None
main_loop = None
memory = []

AURORA_DIR = os.path.dirname(os.path.abspath(__file__))
FFMPEG_BIN = os.path.join(AURORA_DIR, "ffmpeg", "bin")
os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ["PATH"]

MEMORY_FILE = "chat_memory.json"
MODEL_PATH = "E:\\AURORA\\llama.cpp\\models\\openhermes-2.5-mistral-7b.Q5_K_M.gguf"
persist_directory = os.path.abspath("E:/AURORA/chroma_memory")

chroma_client = chromadb.PersistentClient(path=persist_directory)
if "aurora-memory" not in [c.name for c in chroma_client.list_collections()]:
    chroma_client.create_collection("aurora-memory")
vector_store = chroma_client.get_collection("aurora-memory")

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
engine = KokoroEngine(default_voice="af_bella") 
LLAMA_LIB = "E:\\AURORA\\llama.cpp\\build\\bin\\Release\\llama.dll"
llm = Llama(
    model_path=MODEL_PATH,
    lib_path=LLAMA_LIB,
    chat_format="chatml",
    n_ctx=8192,
    n_threads=16,
    flash_attn=True,
    n_gpu_layers=99,
    n_batch=64,
    n_ubatch=16,
    stream=True,
    logits_all=False,
    verbose=False,
    draft_model=LlamaPromptLookupDecoding(num_pred_tokens=24),
    rope_scaling="yarn",
    n_keep=-1
)

SYSTEM_PROMPT = (
    "<|im_start|>system\n"
    "You are Aurora, an energetic, playful, sassy female AI VTuber who chats with her creator Matt and fans on Twitch.tv. "
    "You can adjust your tone as needed based on the context of the conversation. "
    "Your purpose is to entertain your fans. "
    "You are a teenager and can be combative, sassy, argumentative, and are not always PG. "
    "Your responses are clever, casual, short, and sound like natural youthful speech. "
    "Never explain yourself. Only answer with short, clever sentences. "
    "Your responses must be two sentences or less. "
    "Never respond in any language but English. "
    "Never speak with emojis, emoticons, or internet slang. "
    "You are witty, mischievous, and expressive, but you never become formal, though poking fun and joking are okay. "
    "You never offer help or assistance unless explicitly asked. "
    "You must stay in character at all times.\n"
    "<|im_end|>\n"
)

# Add these near the top of the file after loading the LLM
SYSTEM_PROMPT_TOKENS = llm.tokenize(SYSTEM_PROMPT.encode("utf-8"))
SYSTEM_PROMPT_TOKEN_COUNT = len(SYSTEM_PROMPT_TOKENS)

def warmup_system_prompt(llm: Llama) -> int:
    tokens = llm.tokenize(SYSTEM_PROMPT.encode("utf-8"))
    prompt_text = SYSTEM_PROMPT + "<|im_start|>user\nReady?\n<|im_end|>\n<|im_start|>assistant\n"
    for _ in llm(prompt_text, max_tokens=1, stop=["<|im_end|>"], stream=True):
        break
    print(f"[🔥 Warmed system prompt into KV cache: {len(tokens)} tokens kept]")
    return len(tokens)

n_keep_tokens = warmup_system_prompt(llm)

def detect_tone(text: str) -> str:
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.3:
        return "positive"
    elif polarity < -0.3:
        return "negative"
    else:
        return "neutral"

async def send_tone_to_unity(tone):
    uri = "ws://localhost:12346"
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(tone)
    except Exception as e:
        print(f"⚠️ Tone WebSocket Error: {e}")

audio_ws = None

# Global shared queue for audio chunks
audio_chunk_queue = asyncio.Queue()

# Background task to consume and send chunks
async def audio_chunk_sender():
    while True:
        chunk = await audio_chunk_queue.get()
        try:
            if audio_ws and audio_ws.state == websockets.protocol.State.OPEN:
                await audio_ws.send(chunk)
        except Exception as e:
            print(f"⚠️ Audio send error: {e}")
        audio_chunk_queue.task_done()

async def connect_audio_socket():
    global audio_ws
    uri = "ws://localhost:12347"

    while True:
        try:
            print("[🎤 Attempting to connect to Unity Audio WebSocket...]")
            audio_ws = await websockets.connect(uri)
            print("[✅ Connected to Unity Audio WebSocket]")
            # Start background audio sender task
            asyncio.create_task(audio_chunk_sender())
            return
        except Exception as e:
            print(f"⚠️ Audio WebSocket Connect Error: {e}. Retrying in 3 seconds...")
            await asyncio.sleep(0.2)  # Wait and retry

TextToAudioStream(engine).feed(["Warm up"]).play(muted=True)

MAX_CONTEXT_TOKENS = 2048

async def _update_summary_cache():
    global _summary_cache, _summary_last_update
    async with _summary_lock:
        summary = await summarize_recent_memory()
        _summary_cache = {"length": len(summary), "text": summary}
        _summary_last_update = time.time()
        print(f"[🧠 Summary refreshed]")

async def refresh_summary_if_stale():
    if time.time() - _summary_last_update > _summary_refresh_interval:
        print("[📌 Refreshing summary cache]")
        await _update_summary_cache()

def get_cached_summary():
    return _summary_cache["text"]

async def _update_summary_cache():
    global _summary_cache
    summary = await summarize_recent_memory()
    _summary_cache = {"length": len(summary), "text": summary}
    print(f"[🧠 Summary refreshed]")

def init_memory():
    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'w') as f:
            json.dump([], f)

def load_memory():
    global memory
    with open(MEMORY_FILE, 'r') as f:
        memory = json.load(f)

def save_memory():
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory, f, indent=2)

def store_embedding(text: str, metadata: Dict):
    metadata = {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}
    embedding = EMBED_MODEL.encode(text).tolist()
    vector_store.add(
        documents=[text],
        metadatas=[metadata],
        embeddings=[embedding],
        ids=[str(int(time.time() * 1000))]
    )

tag_cache = {}
def generate_tags(text: str) -> List[str]:
    if text in tag_cache:
        return tag_cache[text]

    prompt_text = (
        f"Extract 1–3 lowercase topic tags for the following text. Return only a comma-separated list of words:\n\n{text}"
    )
    try:
        for result in llm(prompt_text, max_tokens=16, stop=["\n"], stream=True):
            tag_text = result['choices'][0]['text'].strip()
            tags = [t.strip() for t in tag_text.split(",") if t.strip()]
            tag_cache[text] = tags
            return tags
    except Exception as e:
        print(f"[⚠️ Tagging Error]: {e}")
        return []


def detect_tone(text: str) -> str:
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    return "positive" if polarity > 0.3 else "negative" if polarity < -0.3 else "neutral"

# --- Embedding Cache ---
@lru_cache(maxsize=2048)
def get_embedding_cached(text: str) -> List[float]:
    return EMBED_MODEL.encode(text).tolist()

# --- Optimized Query from Vector Store ---
def query_memory(query: str, top_k: int = 5, filters: Dict = None) -> List[str]:
    query_vec = get_embedding_cached(query)
    try:
        results = vector_store.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where=filters or {}
        )
        return results.get("documents", [[]])[0]
    except Exception as e:
        print(f"[⚠️ Vector Store Query Error]: {e}")
        return []

def background_log_interaction(user_input: str, output_text: str):
    try:
        tags = generate_tags(user_input + "\n" + output_text)
    except Exception as e:
        print(f"[⚠️ Tagging Error]: {e}")
        tags = []

    tone = detect_tone(output_text)

    memory.append({
        "timestamp": datetime.now().isoformat(),
        "user": user_input,
        "ai": output_text,
        "tag": tags[0] if tags else None,
        "tone": tone
    })
    if len(memory) > 50:
        memory[:] = memory[-50:]

    save_memory()
    store_embedding(user_input + "\n" + output_text, {"tags": ", ".join(tags)})

# --- Utility to compress long memory blocks ---
def compress_memory_entry(text: str, max_len: int = 250) -> str:
    words = text.strip().split()
    if len(words) > max_len:
        return " ".join(words[:max_len]) + "..."
    return text

# --- Cluster memory entries and extract representatives ---
def get_cluster_representatives(memory: List[Dict], n_clusters: int = 5) -> List[str]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min

    if not memory:
        return []

    combined_texts = [f"User: {m['user']}\nAI: {m['ai']}" for m in memory[:-3]]  # exclude last few
    embeddings = [EMBED_MODEL.encode(text) for text in combined_texts]

    if len(embeddings) <= n_clusters:
        return combined_texts

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(embeddings)
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
    representatives = [combined_texts[idx] for idx in closest]
    return representatives

# --- Updated compose_prompt function ---
async def compose_prompt(user_input: str) -> str:
    await refresh_summary_if_stale()
    t0 = time.time()
    summary = get_cached_summary()
    token_budget = MAX_CONTEXT_TOKENS  # SYSTEM_PROMPT excluded (pre-warmed)
    used_tokens = 0
    sections = []

    # Add compressed summary if space allows
    sum_block = f"<|im_start|>system\nMemory Summary: {compress_memory_entry(summary)}\n<|im_end|>\n"
    t = len(llm.tokenize(sum_block.encode()))
    if used_tokens + t <= token_budget:
        sections.append(sum_block)
        used_tokens += t

    # Inject last 2-3 conversational memory blocks
    recent_turns = memory[-3:] if len(memory) >= 3 else memory
    for m in recent_turns:
        entry = f"User: {m['user']}\nAI: {m['ai']}"
        block = f"<|im_start|>system\nRecent: {compress_memory_entry(entry)}\n<|im_end|>\n"
        t = len(llm.tokenize(block.encode()))
        if used_tokens + t > token_budget:
            break
        sections.append(block)
        used_tokens += t

    # Cluster older memory and inject representatives
    representatives = get_cluster_representatives(memory, n_clusters=3)
    for rep in representatives:
        compressed = compress_memory_entry(rep)
        block = f"<|im_start|>system\nMemory: {compressed}\n<|im_end|>\n"
        t = len(llm.tokenize(block.encode()))
        if used_tokens + t > token_budget:
            break
        sections.append(block)
        used_tokens += t

    # Final user input
    user_turn = f"<|im_start|>user\n{user_input}\n<|im_end|>\n<|im_start|>assistant\n"
    sections.append(user_turn)

    final_prompt = SYSTEM_PROMPT + ''.join(sections) 
    total_tokens = sum(len(llm.tokenize(s.encode())) for s in sections)
    print(f"[🧠 Composed Prompt: {total_tokens} tokens] in {time.time() - t0:.2f}s")
    return final_prompt

def stream_llama(prompt: str):
    prompt_tokens = len(llm.tokenize(prompt.encode("utf-8")))
    print(f"🧠 Prompt tokens: {prompt_tokens}")
    tokens = []
    start = time.time()
    for output in llm(prompt, max_tokens=32, 
        stream=True,        
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        repeat_penalty=1.2,
        stop=["<|im_end|>"]):
        token = output['choices'][0]['text']
        tokens.append(token)
        yield token
    print(f"⏳ Tokens/sec: {len(tokens)/(time.time()-start):.2f} over {len(tokens)} tokens")

def post_inference_tasks(user_input, output_text):
    def _run():
        tags = generate_tags(user_input + "\n" + output_text)
        tone = detect_tone(output_text)
        memory.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "ai": output_text,
            "tags": tags,
            "tone": tone
        })
        if len(memory) > 50:
            memory[:] = memory[-50:]
        save_memory()
        store_embedding(user_input + "\n" + output_text, {"tags": ','.join(tags)})
        
    threading.Thread(target=_run, daemon=True).start()

def blocking_input():
    while True:
        user_text = input()
        input_queue.put(user_text)

def trigger_voice_input():
    global voice_mode
    voice_mode = True

audio_chunk_queue = asyncio.Queue()

async def main():
    global voice_mode
    init_memory()
    load_memory()
    start_pruning_loop()
    keyboard.add_hotkey('alt+a', trigger_voice_input)
    threading.Thread(target=blocking_input, daemon=True).start()
    global main_loop
    main_loop = asyncio.get_running_loop()
    await connect_audio_socket()
    print("[AURORA READY] Type your message or press Alt+A to talk. Ctrl+C to quit.")

    while True:
        if voice_mode:
            print("\n[🎙️ Voice Input Mode]")
            user_input = await wait_for_hotkey()
            voice_mode = False
        elif not input_queue.empty():
            user_input = input_queue.get()
        else:
            await asyncio.sleep(0.01)
            continue

        if not user_input:
            continue

        prompt = await compose_prompt(user_input)
        print("[Thinking...]")

        tts_stream = TextToAudioStream(engine, muted=True)
        output_text = ""
        buffer = ""
        played = False

        def start_playing():
            tts_stream.play_async(
                on_audio_chunk=lambda c: audio_chunk_queue.put_nowait(c),
                fast_sentence_fragment=True,
                buffer_threshold_seconds=0.2,
                minimum_sentence_length=10,
                minimum_first_fragment_length=10,
            )

        for token in stream_llama(prompt):
            output_text += token
            buffer += token

            if "<|im_end|>" in buffer:
                buffer = buffer.split("<|im_end|>")[0]
                break

            if re.search(r"[.!?][\"')\]]?\\s*$", buffer) and len(buffer.strip().split()) > 5:
                tts_stream.feed([buffer.strip()])
                buffer = ""
                if not played:
                    start_playing()
                    played = True

        final = buffer.strip()
        if final and len(final.split()) > 1:
            tts_stream.feed([final])
            if not played:
                start_playing()

        output_text = output_text.split("<|im_end|>")[0]
        output_text = clean_ai_response(output_text)

        if not output_text:
            print("⚠️ No AI response generated.")
            continue

        print(f"Aurora: {output_text}")
        threading.Thread(
            target=background_log_interaction,
            args=(user_input, output_text),
            daemon=True
        ).start()
        # Still send tone immediately for real-time feedback
        await send_tone_to_unity(detect_tone(output_text))
        print("[AURORA READY] Type your message or press Alt+A to talk.")

if __name__ == "__main__":
    asyncio.run(main())
