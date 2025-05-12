# file: aurora_tts.py

import os
import sys
import string
import random
from RealtimeTTS import TextToAudioStream, KokoroEngine

# 🗣️ Voice setup
DEFAULT_VOICE = "af_heart"  # or your af_bella equivalent
PREWARM_TEXT = "Warming up!"

# Initialize Kokoro engine
engine = KokoroEngine(default_voice=DEFAULT_VOICE, debug=False)

# Prewarm engine (optional but helps)
def prewarm():
    print(f"Prewarming {DEFAULT_VOICE}...")
    TextToAudioStream(engine).feed([PREWARM_TEXT]).play(muted=True)

def on_word_callback(word):
    """Optional: Called for every spoken word (for English voices)"""
    if word.word not in set(string.punctuation):
        print(f"{word.word}", end=" ", flush=True)

def speak(text: str):
    """Speak a text string using streaming TTS."""
    speed = max(0.1, 1.0 + random.uniform(-0.4, 0.8))  # vary a little naturally
    engine.set_voice(DEFAULT_VOICE)
    engine.set_speed(speed)

    print(f"[🗣️ Speaking at speed {speed:.2f}]")
    TextToAudioStream(engine, on_word=on_word_callback).feed([text]).play(log_synthesized_text=True)

def shutdown():
    """Gracefully shut down the TTS engine."""
    print("[🛑 Shutting down TTS engine...]")
    engine.shutdown()

# 🛑 No code runs on import
