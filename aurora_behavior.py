# file: aurora_behavior.py
from llama_cpp import Llama
import json
import asyncio
import re

llm = Llama(model_path="E:\\MISTRAL_TRAIN\\aurora_behavior_q4.gguf", n_ctx=1024, n_threads=12, n_layers=-1)

instruction_prefix = (
    "You are an intent-to-animation model. "
    "Given a short user phrase, return a JSON object describing:\n"
    "- intent\n- locomotion: {VelocityX, VelocityY, StandStyle, LocomotionMode}\n"
    "- gesture: {ArmBlendH, ArmBlendV}\n\n"
    "Phrase: \"I'm feeling cute.\"\n"
    "Output: {\"intent\": \"pose\", \"locomotion\": {\"VelocityX\": 0.3, \"VelocityY\": 0.0, \"StandStyle\": 0.3, \"LocomotionMode\": 0}, \"gesture\": {\"ArmBlendH\": 0.3, \"ArmBlendV\": -0.98}}\n\n"
)

# Expected keys in output JSON
EXPECTED_KEYS = {"intent", "locomotion", "gesture"}

def extract_and_fix_json(text: str):
    match = re.search(r"\{.*", text, re.DOTALL)
    if not match:
        return None

    raw_json = match.group(0)

    # Fix braces
    open_count = raw_json.count('{')
    close_count = raw_json.count('}')
    while close_count < open_count:
        raw_json += '}'
        close_count += 1

    try:
        data = json.loads(raw_json)
        # Ensure required fields
        if "locomotion" not in data:
            data["locomotion"] = {
                "VelocityX": 0.0,
                "VelocityY": 0.0,
                "StandStyle": 0.0,
                "LocomotionMode": 0
            }
        if "gesture" not in data:
            data["gesture"] = {
                "ArmBlendH": 0.0,
                "ArmBlendV": 0.0
            }
        if "intent" not in data:
            data["intent"] = "idle"
        return data
    except json.JSONDecodeError:
        return None

def validate_json(output_json):
    if not isinstance(output_json, dict):
        return False
    return EXPECTED_KEYS.issubset(output_json.keys())      
      
async def run_phi2_behavior_agent(prompt: str) -> dict:
    full_prompt = f"{prompt.strip()}\n\n### Instruction: Return behavior as JSON:"
    output = llm(
            f"{instruction_prefix}Phrase: \"{prompt}\"\nOutput:", 
            max_tokens=128, 
            stop=None
        )
    text = output["choices"][0]["text"]

    json_str = text.strip().split("}")[0] + "}"
    parsed = extract_and_fix_json(json_str)
    if parsed and validate_json(parsed):
        return json.dumps(parsed, indent=2)
    else:
        print("❌ Failed to parse valid JSON response.")
