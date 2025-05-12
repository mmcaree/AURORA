import json
import subprocess

def format_prompt(user_input):
    return f"""### Instruction:
Aurora is an expressive 3D VTuber. Based on the following user input, generate a behavior control packet in JSON.

### Input:
{user_input}

### Response:
"""


def generate_behavior_packet_llm(user_input, model_cmd="./main -m ./models/phi2.gguf"):
    prompt = format_prompt(user_input)
    result = subprocess.run(f'{model_cmd} -p "{prompt}"', shell=True, capture_output=True, text=True)
    output = result.stdout
    try:
        json_start = output.index("{")
        json_obj = json.loads(output[json_start:])
        return json_obj
    except Exception as e:
        print("⚠️ Failed to parse model output:", e)
        print("Raw output:", output)
        return None


if __name__ == "__main__":
    user_text = input("Aurora prompt: ")
    behavior = generate_behavior_packet_llm(user_text)
    print("\n🎭 Behavior Packet:\n", json.dumps(behavior, indent=2))
