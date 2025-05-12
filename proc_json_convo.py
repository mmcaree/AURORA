import os
import json

# CONFIG
input_folder = r"C:\Users\mmcar\Desktop\ai_files\json_files"  # Folder with your diarization JSON files
output_file = "evil.txt"

all_texts = []  

# Load all files
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        file_path = os.path.join(input_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            segments = data.get("output", {}).get("segments", [])
            for seg in segments:
                text = seg.get("text", "").strip()
                if text:
                    all_texts.append(text)

print(f"Found {len(all_texts)} text lines.")

# Save all texts to a simple file
with open(output_file, 'w', encoding='utf-8') as out_f:
    for line in all_texts:
        out_f.write(line + "\n")

print(f"✅ All sentences written to {output_file}")
