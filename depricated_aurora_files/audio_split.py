import json
import re

input_file = "C:\Users\mmcar\Desktop\ai_files\ai_ethics.txt"
output_file = "C:\Users\mmcar\Desktop\ai_files\ai_ethics.jsonl"

def clean_text(text):
    # Light cleaning
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

examples = []
messages = []

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    if not line:
        continue
    if line.lower().startswith("user:"):
        if messages:
            examples.append({"messages": messages})
            messages = []
        user_text = clean_text(line[5:])
        messages.append({"role": "user", "content": user_text})
    elif line.lower().startswith("aurora:"):
        aurora_text = clean_text(line[7:])
        messages.append({"role": "assistant", "content": aurora_text})

if messages:
    examples.append({"messages": messages})

# Save as JSONL
with open(output_file, 'w', encoding='utf-8') as out_f:
    for ex in examples:
        out_f.write(json.dumps(ex) + "\n")

print(f"Done! {len(examples)} examples written to {output_file}")
