# file: aurora_memory_viewer.py
import os
import json
from flask import Flask, render_template_string, request, redirect, url_for

MEMORY_FILE = "chat_memory.json"

app = Flask(__name__)

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Aurora's Memory Viewer</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; background: #f4f4f4; }
        h1 { color: #444; }
        .entry { background: white; padding: 10px 15px; margin-bottom: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .meta { color: #888; font-size: 0.9em; margin-top: 4px; }
        form { display: inline; }
    </style>
</head>
<body>
    <h1>🧠 Aurora's Long-Term Memory</h1>
    {% for item in memory %}
        <div class="entry">
            <div><strong>You:</strong> {{ item.user }}</div>
            <div><strong>Aurora:</strong> {{ item.ai }}</div>
            <div class="meta">{{ item.timestamp }} | {{ item.window }} | tone: {{ item.tone }} | tag: {{ item.tag }}</div>
            <form method="post" action="/delete">
                <input type="hidden" name="timestamp" value="{{ item.timestamp }}">
                <button type="submit">🗑️ Delete</button>
            </form>
        </div>
    {% endfor %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    with open(MEMORY_FILE, "r") as f:
        memory = json.load(f)
    return render_template_string(TEMPLATE, memory=memory[::-1])

@app.route("/delete", methods=["POST"])
def delete():
    timestamp = request.form["timestamp"]
    with open(MEMORY_FILE, "r") as f:
        memory = json.load(f)
    memory = [m for m in memory if m.get("timestamp") != timestamp]
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
