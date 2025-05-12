# file: onnx_speak.py
import onnxruntime as ort
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import os
from misaki import en
import asyncio
import websockets

# Initialize once
g2p = en.G2P(trf=False, british=False, fallback=None)
# 📂 Model path
MODEL_PATH = "E:/AURORA/kokoro/model_q8f16.onnx"
OUTPUT_WAV = "E:/AURORA/piper/output.wav"

# 🎚️ Voice control params
TEMPERATURE = 0.7  # 0.0 = deterministic, 1.0 = creative
NOISE_SCALE = 0.5  # 0.0 = perfect, 1.0 = noisy/playful
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

# 🧠 Load the ONNX model (only once)
session = ort.InferenceSession(MODEL_PATH)

def text_to_tokens(text):
    tokens = []
    for c in text:
        tokens.append(VOCAB.get(c, 16))
    return tokens

async def synthesize(text: str):
    phonemes, _ = g2p(text)
    tokens = text_to_tokens(phonemes)
    if len(tokens) > 510:
        tokens = tokens[:510]

    voices = np.fromfile('E:/AURORA/kokoro/af_bella.bin', dtype=np.float32).reshape(-1, 1, 256)
    ref_s = voices[len(tokens)]  # Style vector based on text length

    tokens = [[0, *tokens, 0]]
    # before running session
    inputs = {
        "input_ids": tokens,
        "style": ref_s,   # Use style index 0
        "speed": np.ones(1, dtype=np.float32)
    }


    output = session.run(None, inputs)[0]


    audio = output[0].squeeze()  # (samples,)
    audio = np.clip(audio, -1, 1)  # ensure safe range

    # Save to WAV
    wavfile.write(OUTPUT_WAV, 24000, (audio * 32767).astype(np.int16))
    print(f"[✅ Saved]: {OUTPUT_WAV}")
    
    #send phonemes to Unity model
    #asyncio.create_task(send_phonemes(phonemes))
    #play audio (now sending to Unity to handle)
    #play_audio(OUTPUT_WAV)
    
async def notify_unity_audio_ready():
    uri = "ws://localhost:12347"  # New WebSocket for Unity notifications
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send("new_audio_ready")
    except Exception as e:
        print(f"⚠️ WebSocket error: {e}")

def play_audio(path: str):
    """Play a WAV audio file."""
    if not os.path.exists(path):
        print(f"⚠️ Audio file not found: {path}")
        return
    samplerate, data = wavfile.read(path)
    sd.play(data, samplerate)
    sd.wait()


def speak(text: str):
    asyncio.run(synthesize(text))
    asyncio.run(notify_unity_audio_ready())

# 🛑 No executable code at bottom
