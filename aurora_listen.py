# file: aurora_listen.py
import sounddevice as sd
from scipy.io.wavfile import write
import subprocess
import os
import keyboard
import threading
import re
import win32gui
import win32process
import psutil
import numpy as np
from onnxruntime import InferenceSession
import time
import json
from queue import Queue
import collections
from RealtimeSTT import AudioToTextRecorder

WHISPER_PATH = "E:\\AURORA\\whisper.cpp\\build\\bin\\Release\\whisper-cli.exe"
WHISPER_MODEL = "E:\\AURORA\\whisper.cpp\\models\\ggml-base.en.bin"
MIC_WAV = "E:\\AURORA\\microphone.wav"
MEMORY_FILE = "chat_memory.json"

VOICE_NAME = "af_bella"
MODEL_PATH = "E:/AURORA/kokoro/model_q8f16.onnx"
VOICES_PATH = "E:/AURORA/kokoro"
OUTPUT_WAV = "E:/AURORA/piper/converted_output.wav"

# Initialize ONNX model session
sess = InferenceSession(MODEL_PATH)
voices = np.fromfile(f"{VOICES_PATH}/{VOICE_NAME}.bin", dtype=np.float32).reshape(-1, 1, 256)

# Vocabulary mapping
VOCAB = {
    ";": 1,
    ":": 2,
    ",": 3,
    ".": 4,
    "!": 5,
    "?": 6,
    "—": 9,
    "…": 10,
    "\"": 11,
    "(": 12,
    ")": 13,
    "“": 14,
    "”": 15,
    " ": 16,
    "\u0303": 17,
    "ʣ": 18,
    "ʥ": 19,
    "ʦ": 20,
    "ʨ": 21,
    "ᵝ": 22,
    "\uAB67": 23,
    "A": 24,
    "I": 25,
    "O": 31,
    "Q": 33,
    "S": 35,
    "T": 36,
    "W": 39,
    "Y": 41,
    "ᵊ": 42,
    "a": 43,
    "b": 44,
    "c": 45,
    "d": 46,
    "e": 47,
    "f": 48,
    "h": 50,
    "i": 51,
    "j": 52,
    "k": 53,
    "l": 54,
    "m": 55,
    "n": 56,
    "o": 57,
    "p": 58,
    "q": 59,
    "r": 60,
    "s": 61,
    "t": 62,
    "u": 63,
    "v": 64,
    "w": 65,
    "x": 66,
    "y": 67,
    "z": 68,
    "ɑ": 69,
    "ɐ": 70,
    "ɒ": 71,
    "æ": 72,
    "β": 75,
    "ɔ": 76,
    "ɕ": 77,
    "ç": 78,
    "ɖ": 80,
    "ð": 81,
    "ʤ": 82,
    "ə": 83,
    "ɚ": 85,
    "ɛ": 86,
    "ɜ": 87,
    "ɟ": 90,
    "ɡ": 92,
    "ɥ": 99,
    "ɨ": 101,
    "ɪ": 102,
    "ʝ": 103,
    "ɯ": 110,
    "ɰ": 111,
    "ŋ": 112,
    "ɳ": 113,
    "ɲ": 114,
    "ɴ": 115,
    "ø": 116,
    "ɸ": 118,
    "θ": 119,
    "œ": 120,
    "ɹ": 123,
    "ɾ": 125,
    "ɻ": 126,
    "ʁ": 128,
    "ɽ": 129,
    "ʂ": 130,
    "ʃ": 131,
    "ʈ": 132,
    "ʧ": 133,
    "ʊ": 135,
    "ʋ": 136,
    "ʌ": 138,
    "ɣ": 139,
    "ɤ": 140,
    "χ": 142,
    "ʎ": 143,
    "ʒ": 147,
    "ʔ": 148,
    "ˈ": 156,
    "ˌ": 157,
    "ː": 158,
    "ʰ": 162,
    "ʲ": 164,
    "↓": 169,
    "→": 171,
    "↗": 172,
    "↘": 173,
    "ᵻ": 177
  }

def text_to_tokens(text):
    tokens = []
    for c in text:
        tokens.append(VOCAB.get(c, 16))
    return tokens

def speak(text):
    print(f"[DEBUG] Speaking: {text}")
    tokens = text_to_tokens(text)

    if len(tokens) > 510:
        tokens = tokens[:510]

    tokens = [[0, *tokens, 0]]
    ref_index = min(len(tokens[0]), len(voices) - 1)
    ref_s = voices[ref_index]

    audio = sess.run(None, dict(
        input_ids=tokens,
        style=ref_s,
        speed=np.ones(1, dtype=np.float32),
    ))[0]

    audio = np.clip(audio, -1.0, 1.0)
    audio = (audio * 32767).astype(np.int16)
    write(OUTPUT_WAV, 24000, audio)

    os.system(f'start {OUTPUT_WAV}')

def clean_ai_response(response: str) -> str:
    response = response.strip()
    response = re.sub(r'^(Aurora:|AI:)\s*', '', response, flags=re.IGNORECASE)
    response = re.sub(r'\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]', '', response).strip()
    response = re.sub(r'[\[\]]+', '', response)
    response = response.replace("\n", " ").replace("  ", " ")
    return response.strip()

# === Always Listening WebRTC VAD + ASR Thread ===
always_listening = False
frame_duration_ms = 30
sample_rate = 16000
frame_size = int(sample_rate * frame_duration_ms / 1000)
bytes_per_sample = 2
input_queue = Queue()

class Frame:
    def __init__(self, bytes, timestamp):
        self.bytes = bytes
        self.timestamp = timestamp

recorder = None
last_text = ""

def vad_listener_loop():
    global recorder, last_text

    def process_transcription(text):
        global last_text
        cleaned = text.strip()
        if cleaned and cleaned.lower() != last_text.lower():
            print(f"[📝 Recognized]: {cleaned}")
            input_queue.put(cleaned)
            last_text = cleaned
        else:
            print("⚠️ Skipping duplicate or empty input")

    print("[👂 Initializing RealtimeSTT...]")

    recorder = AudioToTextRecorder(
        model="base.en",
        compute_type="int8",
        language="en",
        use_microphone=True,
        webrtc_sensitivity=1,
        post_speech_silence_duration=0.3,
        min_gap_between_recordings=1.0,
        on_realtime_transcription_stabilized=process_transcription,
        debug_mode=True,
    )

    print("[👂 Aurora Always Listening Enabled - RealtimeSTT]")
    recorder.start()

    # 🔁 This loop is needed for .text(...) to fire callbacks
    try:
        while always_listening:
            recorder.text(process_transcription)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("🔇 Stopping RealtimeSTT")
    finally:
        recorder.stop()
        recorder.shutdown()


def start_vad_thread():
    global always_listening
    if always_listening:
        return
    always_listening = True
    threading.Thread(target=vad_listener_loop, daemon=True).start()

def stop_vad():
    global recorder, always_listening
    if recorder:
        recorder.stop()
        recorder.shutdown()
    always_listening = False
    print("👂 VAD Disabled")

def is_listening():
    return always_listening

def wait_for_hotkey():
    global always_listening
    if not always_listening:
        print("👂 Enabling VAD")
        start_vad_thread()
    else:
        print("👂 VAD Disabled")
        always_listening=False
        stop_vad()

def is_terminal_focused():
    try:
        hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd).lower()
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        process = psutil.Process(pid)
        proc_name = process.name().lower()

        return (
            'cmd.exe' in proc_name or 
            'python.exe' in proc_name or 
            'powershell.exe' in proc_name or
            'aurora' in title or
            'powershell' in title or
            'command prompt' in title or
            'terminal' in title
        )
    except Exception:
        return False

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_memory(memory):
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory, f, indent=2)

def prune_memory():
    memory = load_memory()
    if len(memory) <= 1000:
        return

    new_memory = []
    for entry in memory:
        tag = entry.get("tag", "unknown")
        if tag in ("preference", "fact"):
            new_memory.append(entry)
        elif tag == "opinion" and random.random() > 0.3:
            new_memory.append(entry)
        elif tag == "joke" and random.random() > 0.1:
            new_memory.append(entry)
        elif tag == "greeting" and random.random() > 0.2:
            new_memory.append(entry)
    save_memory(new_memory)
    print(f"[🧹 Pruned Memory: {len(memory) - len(new_memory)} entries removed]")

def start_pruning_loop():
    def loop():
        while True:
            prune_memory()
            time.sleep(300)

    threading.Thread(target=loop, daemon=True).start()
