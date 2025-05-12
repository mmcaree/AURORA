import os
import numpy as np
import soundfile as sf
from onnxruntime import InferenceSession
from pathlib import Path

# Constants
VOICE_DIR = Path('voices')
MODEL_DIR = Path('onnx')
OUTPUT_WAV = Path('piper/converted_output.wav')

# Settings
DEFAULT_VOICE = 'af'  # Voice file prefix (e.g., af_bella.bin --> 'af')
MODEL_NAME = 'model_q8f16.onnx'  # Highly efficient model
SAMPLERATE = 24000

# Load Model
model_path = MODEL_DIR / MODEL_NAME
session = InferenceSession(str(model_path))

# Dummy phoneme to ID map (simplified)
# You should replace this with a real phonemizer later
PHONEME_ID_MAP = {
    'a': 14, 'b': 15, 'c': 16, 'd': 17, 'e': 18, 'f': 19,
    'g': 20, 'h': 21, 'i': 22, 'j': 23, 'k': 24, 'l': 25,
    'm': 26, 'n': 27, 'o': 28, 'p': 29, 'q': 30, 'r': 31,
    's': 32, 't': 33, 'u': 34, 'v': 35, 'w': 36, 'x': 37,
    'y': 38, 'z': 39, ' ': 3
}

def text_to_tokens(text: str) -> list:
    """Convert text to token IDs (simplified version)."""
    text = text.lower()
    tokens = [PHONEME_ID_MAP.get(c, 3) for c in text]
    return tokens

def load_voice(voice_name: str = DEFAULT_VOICE) -> np.ndarray:
    """Load a voice style embedding (.bin file)."""
    voice_path = VOICE_DIR / f'{voice_name}.bin'
    voices = np.fromfile(voice_path, dtype=np.float32).reshape(-1, 1, 256)
    return voices[0]  # Pick first style

def generate_audio(text: str, voice_name: str = DEFAULT_VOICE):
    """Generate speech audio from text."""
    tokens = text_to_tokens(text)

    if len(tokens) > 510:
        tokens = tokens[:510]

    tokens = [[0] + tokens + [0]]  # Add pad tokens

    style = load_voice(voice_name)

    inputs = {
        'input_ids': np.array(tokens, dtype=np.int64),
        'style': style,
        'speed': np.ones(1, dtype=np.float32),
    }

    audio = session.run(None, inputs)[0]

    sf.write(OUTPUT_WAV, audio.squeeze(), samplerate=SAMPLERATE)
    print(f"[✅] Generated audio saved to {OUTPUT_WAV}")

def play_audio(file_path=OUTPUT_WAV):
    """Play the generated audio."""
    os.system(f'start {file_path}')

# Example for testing
if __name__ == "__main__":
    text = "hello aurora"
    generate_audio(text, voice_name='af')
    play_audio()
