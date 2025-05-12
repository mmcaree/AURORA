import torch
import whisperx
import json
import re

# CONFIG
input_audio = r"C:\Users\mmcar\Desktop\ai_files\A.wav"  # <-- Just point to your WAV file
output_jsonl = r"C:\Users\mmcar\Desktop\ai_files\A.jsonl"

# Step 1: Transcribe and diarize
def transcribe_with_speakers(audio_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisperx.load_model("medium.en", device, compute_type="float32")


    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio)

    # Diarization
    diarize_model = whisperx.DiarizationPipeline()
    diarize_segments = diarize_model(audio)

    # Assign speakers
    result = whisperx.assign_speakers(result["segments"], diarize_segments)
    return result["segments"]

# Step 2: Build prompt/response pairs
def build_dataset(segments):
    dataset = []
    messages = []
    last_speaker = None

    for seg in segments:
        speaker = seg["speaker"]
        text = seg["text"].strip()
        text = re.sub(r'\s+', ' ', text)  # Clean weird spacing

        if speaker != last_speaker and messages:
            dataset.append({"messages": messages})
            messages = []

        # Assign role
        role = "assistant" if speaker == "SPEAKER_1" else "user"
        messages.append({"role": role, "content": text})

        last_speaker = speaker

    if messages:
        dataset.append({"messages": messages})

    return dataset

# Step 3: Save to JSONL
def save_jsonl(dataset, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")

# MAIN
if __name__ == "__main__":
    print("Transcribing and separating speakers...")
    segments = transcribe_with_speakers(input_audio)

    print("Building dataset...")
    dataset = build_dataset(segments)

    print(f"Saving {len(dataset)} examples to {output_jsonl}...")
    save_jsonl(dataset, output_jsonl)

    print("Done! 🎉 Your dataset is ready.")
