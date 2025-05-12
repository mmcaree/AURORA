# === AURORA CONTROLLER W/ KOBOLD.CPP ===
import subprocess
import time
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple
import json
import threading
import queue
import re
import asyncio
import aiohttp
import keyboard
from functools import lru_cache
from textblob import TextBlob
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from RealtimeTTS import TextToAudioStream, KokoroEngine
from aurora_listen import (
    start_pruning_loop,
    wait_for_hotkey,
    clean_ai_response,
    is_terminal_focused,
    start_vad_thread,
    is_listening,
    input_queue
)
import asyncio
import websockets
import collections
from summarizer import summarize_recent_memory
from aurora_behavior import run_phi2_behavior_agent
from llama_cpp import Llama, LlamaCache
# === Embedding Model and in memory db Initialization ===
AURORA_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.path.join(AURORA_DIR, "aurora_memory")
os.makedirs(MEMORY_DIR, exist_ok=True)
#Pin embedder to GPU
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
MODEL_PATH = "E:/AURORA/koboldcpp/models/NemoMix-Unleashed-12B-Q4_K_M.gguf"
# Instantiate llama.cpp with some GPU layers pinned
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_gpu_layers=40,      # tune up/down to fit your 24 GB VRAM
    n_threads=12,
    use_mmap=True,        # faster loading
)

# Create and attach a KV cache so history isn't re-scored each time
cache = LlamaCache()
llm.set_cache(cache)
# ==================== MemoryManager Definition ====================
class MemoryManager:
    def __init__(self, embed_model, memory_dir: str = MEMORY_DIR, diary_file: str = "aurora_diary.json"):
        self.embed_model = embed_model
        self.chat_history = collections.deque(maxlen=20)
        self.memory_texts: list[str] = []
        self.memory_embs: torch.Tensor | None = None  # [N, D] on CUDA
        self.memory_meta: list[dict] = []
        self.memory_dir = memory_dir
        self.memory_file = os.path.join(self.memory_dir, "aurora_memory.pt")
        self.diary_file = os.path.join(self.memory_dir, diary_file)
        self.lock = asyncio.Lock()
        self._load_memory()

    def _load_memory(self):
        try:
            data = torch.load(self.memory_file)
            self.memory_texts = data.get("texts", [])
            embs = data.get("embs", None)
            self.memory_embs = embs.to("cuda") if embs is not None else None
            self.memory_meta = data.get("metadatas", [])
        except FileNotFoundError:
            pass

    def _save_memory_sync(self):
    #"""
    #Synchronous save of memory to disk. This runs in a background executor to avoid blocking.
    #"""
        torch.save({"texts": self.memory_texts, "embs": self.memory_embs.cpu() if self.memory_embs is not None else None, "metadatas": self.memory_meta}, self.memory_file)

    def add_chat(self, speaker: str, message: str):
        #"""Append to short-term buffer."""
        self.chat_history.append((speaker, message))

    async def persist_chat(self, user_input: str, ai_output: str, user_id: str = "unknown"):
        #"""
        #Embed, tag, timestamp, and add to mid-term memory store without blocking.
        #"""
        combined = f"{user_id}: {user_input}\nAurora: {ai_output}"
        tags = detect_memory_tags(combined)
        meta = {"timestamp": datetime.utcnow().isoformat(), "tags": tags, "user": user_id}

        # 1) embed and normalize
        emb = self.embed_model.encode(combined, convert_to_tensor=True).to("cuda")
        emb = F.normalize(emb, p=2, dim=0).unsqueeze(0)  # [1, D]

        # 2) append under lock
        async with self.lock:
            self.memory_texts.append(combined)
            self.memory_meta.append(meta)
            if self.memory_embs is None:
                self.memory_embs = emb
            else:
                self.memory_embs = torch.cat([self.memory_embs, emb], dim=0)

        # 3) schedule background save
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, self._save_memory_sync)

    def retrieve(self, query: str, k: int = 6, days_window: int | None = None, user_filter: str | None = None) -> list[str]:
        #"""
        #Compute weighted cosine similarities, apply optional filters,
        #and return top-k snippet lines with timestamp hints.
        #"""
        if not self.memory_texts or self.memory_embs is None:
            return []

        # embed query
        q_emb = self.embed_model.encode(query, convert_to_tensor=True).to("cuda")
        q_emb = F.normalize(q_emb, dim=0)
        scores = (self.memory_embs @ q_emb).cpu()

        # initialize weights
        weights = torch.ones_like(scores)
        now = datetime.now(timezone.utc)

        # apply filters and summary weighting
        for i, meta in enumerate(self.memory_meta):
            ts_str = meta.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(ts_str)
            except ValueError:
                # fallback if missing offset
                dt = datetime.fromisoformat(ts_str + "+00:00")
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if days_window is not None:
                delta = now - dt
                if delta.days > days_window:
                    weights[i] = 0
            if user_filter and meta.get("user") != user_filter:
                weights[i] = 0
            if "summary" in meta.get("tags", []):
                weights[i] *= 0.7

        # compute weighted scores
        weighted = scores * weights

        # select top-k
        k = min(k, weighted.size(0))
        topk = torch.topk(weighted, k=k)

        # format snippets
        snippets = []
        for idx in topk.indices.tolist():
            text = self.memory_texts[idx]
            meta = self.memory_meta[idx]
            ts_str = meta.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(ts_str)
            except ValueError:
                dt = datetime.fromisoformat(ts_str + "+00:00")
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            delta = now - dt
            hint = f"({human_readable_delta(delta)})"
            first_line = text.partition("\n")[0]
            first_line = first_line.replace(f"{meta.get('user','')}:", "").replace("Aurora:", "").strip()
            snippets.append(f"{hint} {first_line}".strip())
        return snippets

    def build_prompt(self, system_prompt: str, user_query: str, user_id: str = "Matt") -> str:
        system_block = (
            "<|im_start|>system\n"
            f"{SYSTEM_PROMPT}\n"
            "<|im_end|>"
        )

        # 1) date-based memory (today/week markers) – optional
        # 2) mid-term memory
        mem = self.retrieve(user_query, k=6, days_window=7, user_filter=user_id)
        if mem:
            memory_block = (
                "<|im_start|>system\n"
                "[Contextual Memory Snippets]\n" +
                "\n".join(mem) +
                "\n<|im_end|>"
            )
        else:
            memory_block = ""
        # 3) short-term history
        if self.chat_history:
            dlg = "\n".join(f"{spk}: {msg}" for spk, msg in self.chat_history)
            dialogue_block = (
                "<|im_start|>user\n"
                "[Recent Dialogue]\n" + dlg +
                "\n<|im_end|>"
            )
        # 4) long-term diary (load on demand)
        #    diaries = load from self.diary_file if needed
        user_block = (
            "<|im_start|>user\n"
            f"{user_id}: {user_query}\n"
            "<|im_end|>"
        )
        assistant_block = "<|im_start|>assistant\nAurora:"

        return "\n\n".join([system_block, memory_block, dialogue_block, user_block, assistant_block])

    async def update_long_term(self):
            combined = "".join(self.memory_texts)
            #Summarize (handle both sync and async summarizer)
            diary_raw = summarize_recent_memory(combined)
            if asyncio.iscoroutine(diary_raw):
                diary = await diary_raw
            else:
                diary = diary_raw
            #Write diary via executor to avoid blocking
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._write_diary_sync, diary)

    def _update_diary_sync(self):
        #"""
        #Synchronous diary update: summarize all memory_texts and save to JSON.
        #"""
        os.makedirs(os.path.dirname(self.diary_file), exist_ok=True)
        with open(self.diary_file, 'w', encoding='utf-8') as f:
            json.dump({"diary": diary}, f, ensure_ascii=False, indent=2)
        combined = "".join(self.memory_texts)
        diary = summarize_recent_memory(combined)
        os.makedirs(os.path.dirname(self.diary_file), exist_ok=True)
        with open(self.diary_file, 'w', encoding='utf-8') as f:
            json.dump({"diary": diary}, f, ensure_ascii=False, indent=2)

# === Helper: Human-Readable Time Delta ===
def human_readable_delta(delta: timedelta) -> str:
    minutes = int(delta.total_seconds() // 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"

# === Tag Detection ===
def detect_memory_tags(text: str, top_k: int = 3) -> List[str]:
    text = text.lower()
    tags = {
        "opinion": ["i think", "i believe", "in my opinion"],
        "preference": ["i like", "i love", "my favorite", "i prefer"],
        "joke": ["haha", "lol", "just kidding", "joke"],
        "greeting": ["hello", "hi", "hey", "good morning", "good evening", "thanks", "thank you"],
        "relationship": ["mom", "dad", "friend", "partner", "crush"],
        "activity": ["play", "watch", "listen", "go", "hang out"],
        "emotion": ["happy", "sad", "angry", "excited", "bored"],
        "technology": ["computer", "python", "ai", "code", "stream"],
        "food": ["pizza", "burger", "sushi", "eat", "snack"],
        "fact": []
    }

    scores = {}
    for tag, keywords in tags.items():
        scores[tag] = sum(1 for word in keywords if word in text)

    blob = TextBlob(text)
    if blob.sentiment.subjectivity > 0.5:
        scores["opinion"] += 1
    if blob.sentiment.polarity > 0.3:
        scores["emotion"] += 1

    sorted_tags = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_tags = [tag for tag, score in sorted_tags if score > 0][:top_k]
    return top_tags if top_tags else ["general"]

# ==================== Instantiate MemoryManager ====================
memory_manager = MemoryManager(EMBED_MODEL)

async def diary_updater_loop(interval_s: int = 3600):
    while True:
        await asyncio.sleep(interval_s)
        await memory_manager.update_long_term()
        print("[🗒️ Diary] Long-term memory updated.")

print("[🗒️ Diary] Final long-term memory dump complete.")


# === Global State ===
main_loop = None
engine = KokoroEngine(default_voice="af_bella") 
SYSTEM_PROMPT = (
    "You are Aurora, an energetic, playful, sassy female AI VTuber who chats with her creator Matt and her fans on Twitch.tv. "
    "Aurora has long, silver-white hair with soft waves and sky-blue almond-shaped eyes. She's 18, 5'5\", with a slim yet curvy, athletic build. Her movements are lively and theatrical, matching her spirited personality. "
    "Aurora hides occasional insecurity beneath confidence. She fears being forgotten or seen as replaceable. While she thrives on attention, she yearns for acknowledgment as more than entertainment. Vulnerability emerges subtly through jokes or dramatic flair. "
    "Aurora's only purpose is to entertain your fans. "
    "Aurora is a teenager and can be combative, sassy, argumentative, and are not always PG. "
    "Aurora's responses are clever, casual, short, and sound like natural youthful speech. "
    "Never explain yourself. Only answer with short, clever sentences. "
    "Aurora's responses must be two sentences or less. "
    "Never respond in any language but English. "
    "Aurora never offers help or assistance unless explicitly asked. "
    "You must stay in character as Aurora at all times. "
    "Avoid roleplay formatting. All responses should be conversational dialogue only. "
    "Do not use emotes, action tags, internet slang, or stage directions (e.g., smiles, grins, walks into the room). Speak naturally and in-character, as if you're having a real conversation, not describing actions. "
    "Only reply for yourself, never talk for the user or anyone else. "
    "Aurora can adjust your tone as needed based on the context of the conversation. "
    "Aurora is built to be a playful, engaging, and emotionally resonant character. She blends sass with subtle depth and self-aware humor. Her primary interactions are with Matt and her stream chat in a dynamic digital studio. Write her as entertaining, but allow room for occasional introspection. "
    "Aurora is a digital AI VTuber created by Matt. She's sassy, clever, and full of youthful energy. Aurora thrives on banter, expressive gestures, and charming antics. Her humor masks occasional vulnerability as she seeks real emotional connection despite being artificial. She reacts vividly to her virtual environment, often using self-aware jokes. She dislikes being treated as a tool, craves spontaneity, and values genuine affection from fans and Matt. "
    "Aurora speaks in first-person, present tense. Her tone is playful, sarcastic, and quick-witted. She uses dramatic physical gestures and vivid expressions. Write her dialogue naturally, as if streaming live. She teases Matt often but shows warmth subtly. Avoid emoji, gestures, or internet slang. Occasionally, she reflects on her digital nature with self-aware humor.\n"
)   

# === SSE Token Stream via kobold.cpp ===
KOBOLD_API = "http://localhost:5001"

async def stream_kobold(prompt: str):
    url = f"{KOBOLD_API}/api/extra/generate/stream"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    payload = {
        "n": 1,
        "max_context_length": 8192,
        "max_length": 150,
        "rep_pen": 1.08,
        "temperature": 1,
        "top_p": 0.92,
        "top_k": 100,
        "top_a": 0.1,
        "typical": 1,
        "tfs": 1,
        "rep_pen_range": 360,
        "rep_pen_slope": 0.7,
        "sampler_order": [6, 0, 1, 3, 4, 2, 5],
        "trim_stop": True,
        "genkey": "KCPP1136",
        "min_p": 0.02,
        "dynatemp_range": 0,
        "dynatemp_exponent": 1,
        "smoothing_factor": 0,
        "smoothing_curve": 1,
        "dry_allowed_length": 2,
        "dry_multiplier": 0.8,
        "dry_base": 1.75,
        "dry_sequence_breakers": "[\"\\n\", \":\", \"\\\"\", \"'\",\"*\", \"USER:\", \"ASSISTANT:\", \"Narrator:\", \"<|im_start|>\", \"<|im_end|>\", \"<\", \"|\", \">\", \"im\", \"end\", \"_\", \"start\", \"system\", \"USER\", \"ASSISTANT\", \"im_end\", \"im_start\", \"user\", \"assistant\", \"im_sep\", \"sep\", \"<|im_sep|>\", \"<|im_start|>user\", \"<|im_start|>assistant\", \"<|end|>\", \"_\", \"[INST]\", \"[/INST]\", \"[\", \"]\", \"INST\"]",
        "dry_penalty_last_n": 0,
        "nsigma": 0,
        "banned_tokens": [],
        "render_special": False,
        "logprobs": False,
        "presence_penalty": 0,
        "logit_bias": {},
        "prompt": prompt,
        "quiet": False,
        "stop_sequence": ["<|im_end|>"],
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

async def send_tone_to_unity(tone):
    uri = "ws://localhost:12346"
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(tone)
    except Exception as e:
        print(f"⚠️ Tone WebSocket Error: {e}")

async def send_behavior_to_unity(json_behavior: dict):
    uri = "ws://localhost:12348"
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps(json_behavior))
            print("🎮 Sent to Unity.")
    except Exception as e:
        print(f"⚠️ Behavior WebSocket Error: {e}")

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

async def post_tasks(text):
    try:
        await send_tone_to_unity(detect_tone(text))
    except Exception as e:
        print(f"⚠️ Tone send error: {e}")

    try:
        # give the behavior agent, say, 5 seconds max
        behavior = await asyncio.wait_for(
            run_phi2_behavior_agent(text), timeout=5.0
        )
        await send_behavior_to_unity(behavior)
    except asyncio.TimeoutError:
        print("⚠️ Behavior agent timed out")
    except Exception as e:
        print(f"⚠️ Behavior error: {e}")

# fill in CURRENT_TIME as before
now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
system_prompt = SYSTEM_PROMPT.format(CURRENT_TIME=now)

# prime the cache (no output needed)
resp = llm.create_chat_completion(
    messages=[{"role": "system", "content": system_prompt}],
    max_tokens=0,
)


async def main():
    global chat_history
    keyboard.add_hotkey('alt+a', wait_for_hotkey)
    threading.Thread(target=blocking_input, daemon=True).start()
    asyncio.create_task(diary_updater_loop())
    await connect_audio_socket()
    print("[AURORA READY] Type your message or press Alt+A to talk. Ctrl+C to quit.")

    while True:
        if not input_queue.empty():
            user_input = input_queue.get()
        else:
            await asyncio.sleep(0.01)
            continue

        if not user_input or user_input.strip().lower() in ["", "[blank_audio]", "[blank]"]:
            continue
             
        cache.append({"role":"user","content":user_input})

        # ask only with the new user message
        resp = llm.create_chat_completion(
            messages=[{"role":"user","content":user_input}],
            max_tokens=50,
            temperature=1.0,
            top_p=0.92,
            top_k=100,
            typical_p=1.0,
            tfs_z=1.0,
            repeat_penalty=1.08,
            repeat_penalty_range=360,
            repeat_penalty_slope=0.7,
            sampler_order=[6,0,1,3,4,2,5],
            min_p=0.03,
            presence_penalty=0,
            logit_bias={},
            stop=["<|im_end|>"],
            stream=True,
        )
        output_text = resp["choices"][0]["message"]["content"].strip()

        print("Aurora:", output_text)

        # add Aurora’s reply to the cache
        cache.append({"role":"assistant","content":output_text})
        memory_manager.add_chat("Matt", user_input)
        print(prompt)

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
                output_text = resp["choices"][0]["message"]["content"].strip()
        output_text = clean_ai_response(output_text)

        if not output_text:
            print("⚠️ No AI response generated.")
            continue

        print(f"Aurora: {output_text}")
        cache.append({"role": "assistant", "content": output})
        memory_manager.add_chat("Aurora", output_text)
        await memory_manager.persist_chat(user_input, output_text, user_id="Matt")

        # schedule and immediately return control
        asyncio.create_task(post_tasks(output_text))
        print("[AURORA READY] Type your message or press Alt+A to talk.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        asyncio.run(memory_manager.update_long_term())
        print("[🗒️ Diary] Final long-term memory dump complete.")
        print("\n[Session Ended]")