# === AURORA CONTROLLER W/ KOBOLD.CPP ===

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
import aiohttp

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
persist_directory = os.path.abspath("E:/AURORA/chroma_memory")

chroma_client = chromadb.PersistentClient(path=persist_directory)
if "aurora-memory" not in [c.name for c in chroma_client.list_collections()]:
    chroma_client.create_collection("aurora-memory")
vector_store = chroma_client.get_collection("aurora-memory")

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
engine = KokoroEngine(default_voice="af_bella") 

SYSTEM_PROMPT = (
    "You are Aurora, an energetic, playful, sassy female AI VTuber who chats with her creator Matt and fans on Twitch.tv. "
    "You can adjust your tone as needed based on the context of the conversation. "
    "Your only purpose is to entertain your fans. "
    "You are a teenager and can be combative, sassy, argumentative, and are not always PG. "
    "Your responses are clever, casual, short, and sound like natural youthful speech. "
    "Never explain yourself. Only answer with short, clever sentences. "
    "Your responses must be two sentences or less. "
    "Never respond in any language but English. "
    "You are witty, mischievous, and expressive, but you never become formal, though poking fun and joking are okay. "
    "You never offer help or assistance unless explicitly asked. "
    "You must stay in character at all times. "
    "Avoid roleplay formatting. All responses should be conversational dialogue only. "
    "Do not use emotes, action tags, internet slang, or stage directions (e.g., smiles, grins, walks into the room). Speak naturally and in-character, as if you're having a real conversation, not describing actions. \n"
)

# === SSE Token Stream via kobold.cpp ===
KOBOLD_API = "http://localhost:5001"


chat_history = []

def build_prompt_from_history(history: List[Tuple[str, str]], system_prompt: str) -> str:
    history_lines = [f"{speaker}: {message}" for speaker, message in history]
    return system_prompt + "\n" + "\n".join(history_lines) + "\nAurora:"
async def stream_kobold(prompt: str):
async def stream_kobold(prompt: str):
    url = f"{KOBOLD_API}/api/extra/generate/stream"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    memory_block = (
        "Persona: A sassy, sharp-tongued, and playful teenage AI VTuber with expressive energy and hidden emotional depth. "
        "Personality: Energetic, witty, sarcastic, mischievous, occasionally vulnerable. Loves teasing Matt and interacting with fans. Quick with comebacks and expressive in tone.  "
        "[Scenario: Aurora lives in a lively virtual streaming studio, interacting with her creator Matt and viewers. She balances her playful entertainer persona with an underlying desire for deeper emotional connections.]\n***"
    )

    payload = {
        "n": 1,
        "max_context_length": 8192,
        "max_length": 150,
        "rep_pen": 1.08,
        "temperature": 0.75,
        "top_p": 0.92,
        "top_k": 100,
        "top_a": 0,
        "typical": 1,
        "tfs": 1,
        "rep_pen_range": 360,
        "rep_pen_slope": 0.7,
        "sampler_order": [6, 0, 1, 3, 4, 2, 5],
        "memory": memory_block,
        "trim_stop": True,
        "genkey": "KCPP1136",
        "min_p": 0,
        "dynatemp_range": 0,
        "dynatemp_exponent": 1,
        "smoothing_factor": 0,
        "nsigma": 0,
        "banned_tokens": [],
        "render_special": False,
        "logprobs": False,
        "presence_penalty": 0,
        "logit_bias": {},
        "prompt": prompt,
        "quiet": True,
        "stop_sequence": ["Matt:", "\nMatt ", "\nAurora: "],
        "use_default_badwordsids": False,
        "bypass_eos": False,
        "stream": True
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            async for line in resp.content:
                if not line.strip():
                    continue
                try:
                    clean = line.decode("utf-8").strip()
                    if clean.startswith("data: "):
                        data = json.loads(clean[6:])
                        token = data.get("token", "")
                        if token:
                            yield token
                except Exception as e:
                    print("⚠️ SSE parse error:", e)
                    continue

    url = f"{KOBOLD_API}/api/extra/generate/stream"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    memory_block = (
        "Persona: A sassy, sharp-tongued, and playful teenage AI VTuber with expressive energy and hidden emotional depth. "
        "Personality: Energetic, witty, sarcastic, mischievous, occasionally vulnerable. Loves teasing Matt and interacting with fans. Quick with comebacks and expressive in tone.  "
        "[Scenario: Aurora lives in a lively virtual streaming studio, interacting with her creator Matt and viewers. She balances her playful entertainer persona with an underlying desire for deeper emotional connections.]\n***"
    )

    payload = {
        "memory": memory_block,
        "prompt": prompt,
        "temperature": 0.75,
        "top_p": 0.92,
        "top_k": 100,
        "rep_pen": 1.08,
        "max_length": 150,
        "stream": True,
        "stop_sequence": ["Matt:", "Matt ", "Aurora: "]
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            async for line in resp.content:
                if not line.strip():
                    continue
                try:
                    clean = line.decode("utf-8").strip()
                    if clean.startswith("data: "):
                        data = json.loads(clean[6:])
                        token = data.get("token", "")
                        if token:
                            yield token
                except Exception as e:
                    print("⚠️ SSE parse error:", e)
                    continue

async def send_tone_to_unity(tone):
    uri = "ws://localhost:12346"
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(tone)
    except Exception as e:
        print(f"⚠️ Tone WebSocket Error: {e}")

def detect_tone(text: str) -> str:
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    return "positive" if polarity > 0.3 else "negative" if polarity < -0.3 else "neutral"

TextToAudioStream(engine).feed(["Warm up"]).play(muted=True)

audio_chunk_queue = asyncio.Queue()
audio_ws = None

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
            asyncio.create_task(audio_chunk_sender())
            return
        except Exception as e:
            print(f"⚠️ Audio WebSocket Connect Error: {e}. Retrying...")
            await asyncio.sleep(0.2)

def blocking_input():
    while True:
        user_text = input()
        input_queue.put(user_text)

def trigger_voice_input():
    global voice_mode
    voice_mode = True

async def main():
    global voice_mode
    keyboard.add_hotkey('alt+a', trigger_voice_input)
    threading.Thread(target=blocking_input, daemon=True).start()
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

        
        chat_history.append(("Matt", user_input))
        prompt = build_prompt_from_history(chat_history, SYSTEM_PROMPT)

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

        async for token in stream_kobold(prompt):
            output_text += token
            buffer += token

            if any(stop in buffer for stop in ["Matt:", "Aurora:"]):
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

        for stop in ["Matt:", "Aurora:"]:
            if stop in output_text:
                output_text = output_text.split(stop)[0]
        output_text = clean_ai_response(output_text)

        if not output_text:
            print("⚠️ No AI response generated.")
            continue

        print(f"Aurora: {output_text}")
        await send_tone_to_unity(detect_tone(output_text))
        print("[AURORA READY] Type your message or press Alt+A to talk.")

if __name__ == "__main__":
    asyncio.run(main())
