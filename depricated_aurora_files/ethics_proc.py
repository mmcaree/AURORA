from pathlib import Path

# Load ethics.txt content
ethics_path = Path("E:\\AURORA\\ethics.txt")
lines = ethics_path.read_text(encoding="utf-8").splitlines()

merged_conversations = []
role_buffer = []
current_role = None

def flush_role_buffer():
    if not role_buffer:
        return None
    text = " ".join(role_buffer).strip()
    return {"role": current_role, "content": text}

for line in lines:
    stripped = line.strip()
    if not stripped:
        continue

    role_tag = stripped[-1]
    text = stripped[:-1].strip()

    if role_tag != current_role:
        # if we have a full pair, flush to messages
        flushed = flush_role_buffer()
        if flushed:
            if 'messages' not in locals() or len(messages) == 2:
                messages = []
            messages.append(flushed)
            role_buffer = []

        current_role = role_tag

    role_buffer.append(text)

    if len(locals().get('messages', [])) == 2:
        merged_conversations.append({"messages": messages})
        messages = []

# Flush last buffer
if role_buffer:
    flushed = flush_role_buffer()
    if 'messages' not in locals() or len(messages) == 2:
        messages = []
    messages.append(flushed)
    if len(messages) == 2:
        merged_conversations.append({"messages": messages})

print(merged_conversations[:3])

