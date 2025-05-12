

# file: aurora_controller.py

import subprocess
import time
import os
import sys
from datetime import datetime
from typing import List, Dict
import pygetwindow as gw
import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pytesseract
from PIL import ImageGrab
import threading
from textblob import TextBlob
from RealtimeTTS import TextToAudioStream, KokoroEngine
import keyboard
import re
import queue
from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
import asyncio
import websockets
import numpy as np
from aurora_listen import (
    start_pruning_loop,
    wait_for_hotkey,
    detect_memory_tag,
    clean_ai_response,
    is_terminal_focused
)

global voice_mode
voice_mode = False

input_queue = queue.Queue()

main_loop = None

AURORA_DIR = os.path.dirname(os.path.abspath(__file__))
FFMPEG_BIN = os.path.join(AURORA_DIR, "ffmpeg", "bin")
os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ["PATH"]
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

MEMORY_FILE = "chat_memory.json"
MODEL_PATH = "E:\\AURORA\\llama.cpp\\models\\openhermes-2.5-mistral-7b.Q4_K_M.gguf"
persist_directory = os.path.abspath("E:/AURORA/chroma_memory")

chroma_client = chromadb.PersistentClient(path=persist_directory)
if "aurora-memory" not in [c.name for c in chroma_client.list_collections()]:
    chroma_client.create_collection("aurora-memory")
vector_store = chroma_client.get_collection("aurora-memory")

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
MAX_SCREEN_TEXT_LEN = 1000

engine = KokoroEngine(default_voice="af_bella") 
LLAMA_LIB = "E:\\AURORA\\llama.cpp\\build\\bin\\Release\\llama.dll"
llm = Llama(
    model_path=MODEL_PATH,
    lib_path=LLAMA_LIB,
    chat_format="chatml",
    n_ctx=8192,
    n_threads=12,
    flash_attn = True,
    n_gpu_layers=99,
    n_batch=2048,
    n_ubatch=512,
    stream=True,
    logits_all=False,
    verbose=False,
    rope_scaling="yarn",        # if your llama.cpp build supports it (reduces KV context overhead)
    n_keep=-1,  # ensures the warmed-up system prompt stays in KV cache
    draft_model=LlamaPromptLookupDecoding(num_pred_tokens=16)
)

SYSTEM_PROMPT = (
    "<|im_start|>system\n"
    "You are Aurora, an energetic, playful, sassy female AI VTuber who chats with her creator Matt and fans on Twitch.tv. "
    "Your purpose is to entertain your fans. "
    "You are a teenager and can be combative, sassy, argumentative, and are not always PG. "
    "Your responses are clever, casual, short, and sound like natural youthful speech. "
    "Never explain yourself. Only answer with short, clever sentences. "
    "Your responses must be two sentences or less. "
    "You never respond in any language but English. "
    "You NEVER speak with emojis or internet slang, or describe gestures. "
    "You are witty, mischievous, and expressive, but you never become rude, mean, or formal, though poking fun and joking are okay. "
    "You never offer help or assistance unless explicitly asked. "
    "You must stay in character at all times.\n"
    "<|im_end|>\n"
)

def warmup_system_prompt(llm: Llama) -> int:
    tokens = llm.tokenize(SYSTEM_PROMPT.encode("utf-8"))
    prompt_text = SYSTEM_PROMPT + "<|im_start|>user\nReady?\n<|im_end|>\n<|im_start|>assistant\n"
    for _ in llm(prompt_text, max_tokens=1, stop=["<|im_end|>"], stream=True):
        break
    print(f"[🔥 Warmed system prompt into KV cache: {len(tokens)} tokens kept]")
    return len(tokens)

n_keep_tokens = warmup_system_prompt(llm)


TextToAudioStream(engine).feed(["Warm up"]).play(muted=True)

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

def init_memory():
    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'w') as f:
            json.dump([], f)

def load_memory() -> List[Dict]:
    with open(MEMORY_FILE, 'r') as f:
        return json.load(f)

def save_memory(memory: List[Dict]):
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory, f, indent=2)

def get_active_window_title() -> str:
    try:
        return gw.getActiveWindowTitle() or "Unknown Window"
    except Exception:
        return "Unknown Window"

def store_embedding(text: str, metadata: Dict):
    embedding = EMBED_MODEL.encode(text).tolist()
    doc_id = str(int(time.time() * 1000))
    vector_store.add(
        documents=[text],
        metadatas=[metadata],
        embeddings=[embedding],
        ids=[doc_id]
    )

def capture_screen_text() -> str:
    img = ImageGrab.grab()
    text = pytesseract.image_to_string(img)
    return text.strip()[:MAX_SCREEN_TEXT_LEN]

def summarize_ocr_context(screen_text: str, window: str) -> str:
    prompt = (
        f"[INST] Summarize what the user is doing based on this screen text from {window}:\n"
        f"{screen_text}\n[/INST]"
    )
    result = subprocess.run(
        ["E:\\AURORA\\llama.cpp\\build\\bin\\Release\\llama-cli.exe", "-m", MODEL_PATH, "-p", prompt, "--temp", "0.7", "-n", "150"],
        capture_output=True, text=True
    )
    output = result.stdout.strip()
    for line in output.splitlines():
        if "[/INST]" in line:
            response = line.split("[/INST]", 1)[-1].strip(" </s>[end of text]").strip()
            while response.startswith("AI:"):
                response = response[len("AI:"):].strip()
            return response
    return "[Unclear what the user is doing]"

def store_screen_ocr():
    screen_text = capture_screen_text()
    if not screen_text.strip():
        print("[INFO] No screen text detected.")
        return

    window = get_active_window_title()
    timestamp = datetime.now().isoformat()
    summary = summarize_ocr_context(screen_text, window)
    content = f"Screen summary: {summary}"

    store_embedding(content, {"window": window, "timestamp": timestamp, "tag": "ocr"})
    print(f"[SAVED] Screen OCR Summary from: {window} | {summary}")

def query_memory(query: str, top_k: int = 3, tag_filter: str = None) -> List[str]:
    query_vec = EMBED_MODEL.encode(query).tolist()
    query_args = {"query_embeddings": [query_vec], "n_results": top_k}
    if tag_filter:
        query_args["where"] = {"tag": tag_filter}
    results = vector_store.query(**query_args)
    return results.get("documents", [[]])[0]

MAX_CONTEXT_TOKENS = 2048  # optimize for speed vs quality tradeoff, max 8196

def compose_prompt(user_input: str, memory: List[Dict], window_title: str) -> str:
    token_budget = MAX_CONTEXT_TOKENS - compose_prompt._system_tokens
    prompt_sections = []

    used_tokens = 0

    # === Long-term memory (vector recall) ===
    if vector_store.count() > 0:
        related = query_memory(user_input, top_k=2)
        for m in related:
            compact = f"<|im_start|>system\nMemory: {m.strip()}\n<|im_end|>\n"
            tokens = len(llm.tokenize(compact.encode()))
            if used_tokens + tokens > token_budget:
                break
            prompt_sections.append(compact)
            used_tokens += tokens

    # === Short-term memory (compressed form) ===
    if memory:
        compacted = [
            f"U: {m['user'].strip()} | A: {m['ai'].strip()}" for m in memory[-4:]
        ]
        combined = "<|im_start|>system\n" + " || ".join(compacted) + "\n<|im_end|>\n"
        tokens = len(llm.tokenize(combined.encode()))
        if used_tokens + tokens <= token_budget:
            prompt_sections.append(combined)
            used_tokens += tokens

    # === User turn ===
    user_turn = f"<|im_start|>user\n{user_input}\n<|im_end|>\n<|im_start|>assistant\n"
    user_tokens = len(llm.tokenize(user_turn.encode()))
    if used_tokens + user_tokens <= token_budget:
        prompt_sections.append(user_turn)

    full_prompt = ''.join([compose_prompt._system_prompt] + prompt_sections)
    print(f"[🧠 Composed Prompt: {compose_prompt._system_tokens + used_tokens} tokens]")
    return full_prompt

# Pre-cache system prompt token count
compose_prompt._system_prompt = SYSTEM_PROMPT
compose_prompt._system_tokens = len(llm.tokenize(SYSTEM_PROMPT.encode()))

# Token streaming with timing info
def stream_llama(prompt: str):
    prompt_tokens = len(llm.tokenize(prompt.encode("utf-8")))
    print(f"🧠 Prompt tokens: {prompt_tokens}")

    tokens = []
    start = time.time()

    for output in llm(prompt, max_tokens=32,         
        temperature=0.8,
        top_p=0.95,
        top_k=40,   
        stop=["<|im_end|>"], 
        stream=True):

        token = output['choices'][0]['text']
        tokens.append(token)
        yield token

    duration = time.time() - start
    print(f"⏳ Tokens/sec: {len(tokens)/duration:.2f}")


def blocking_input():
    while True:
        user_text = input()
        input_queue.put(user_text)

def trigger_voice_input():
    global voice_mode
    voice_mode = True

async def main():
    global voice_mode
    init_memory()
    memory = load_memory()
    start_pruning_loop()
    keyboard.add_hotkey('alt+a', trigger_voice_input)
    threading.Thread(target=blocking_input, daemon=True).start()
    global main_loop
    main_loop = asyncio.get_running_loop()
    await connect_audio_socket()
    print("[AURORA READY] Type your message or press Alt+A to talk. Ctrl+C to quit.")

    while True:
        try:
            if voice_mode:
                print("\n[🎙️ Switching to voice input...]")
                user_input = await wait_for_hotkey()
                voice_mode = False
            elif not input_queue.empty():
                user_input = input_queue.get()
            else:
                await asyncio.sleep(0.01)
                continue

            if not user_input:
                continue

            window_title = get_active_window_title()
            prompt = compose_prompt(user_input, memory, window_title)
            print("[Thinking...]")

            tts_stream = TextToAudioStream(engine, muted=True)
            output_text = ""
            buffer = ""

            def start_playing():
                tts_stream.play_async(
                    on_audio_chunk=lambda chunk: audio_chunk_queue.put_nowait(chunk),
                    fast_sentence_fragment=True,
                    fast_sentence_fragment_allsentences=True,
                    fast_sentence_fragment_allsentences_multiple=True,
                    buffer_threshold_seconds=0.2,
                    minimum_sentence_length=10,
                    minimum_first_fragment_length=10,
                )

            played = False
            for token in stream_llama(prompt):
                output_text += token
                buffer += token

                if "<|im_end|>" in buffer:
                    buffer = buffer.split("<|im_end|>")[0]
                    break

                # Sentence boundary detected (basic regex) and long enough
                if re.search(r"[.!?][\"')\]]?\s*$", buffer) and len(buffer.strip().split()) > 5:
                    tts_stream.feed([buffer.strip()])
                    buffer = ""

                    if not played:
                        start_playing()
                        played = True

            # Flush any remaining meaningful buffer
            final = buffer.strip()
            if final and len(final.split()) > 1:
                tts_stream.feed([final])
                if not played:
                    start_playing()


            output_text = output_text.split("<|im_end|>")[0]
            output_text = clean_ai_response(output_text)

            if not output_text:
                print("⚠️ No AI response generated, skipping.")
                continue

            print(f"Aurora: {output_text}")
            tag = detect_memory_tag(output_text)
            tone = detect_tone(output_text)

            await send_tone_to_unity(tone)

            memory.append({
                "timestamp": datetime.now().isoformat(),
                "user": user_input,
                "ai": output_text,
                "window": window_title,
                "tag": tag,
                "tone": tone
            })
            if len(memory) > 50:
                memory = memory[-50:]
            save_memory(memory)
            store_embedding(user_input + "\n" + output_text, {"window": window_title, "tag": tag})

            voice_mode = False
            user_input = None
            print("[AURORA READY] Type your message or press Alt+A to talk.")

        except KeyboardInterrupt:
            print("\n[Session Ended]")
            break


if __name__ == "__main__":
    asyncio.run(main())