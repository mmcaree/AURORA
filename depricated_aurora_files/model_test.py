import time
from llama_cpp import Llama
import os
MODEL_PATH = "E:\\AURORA\\llama.cpp\\models\\openhermes-2.5-mistral-7b.Q4_K_M.gguf"
llm = Llama(
    model_path=MODEL_PATH,
    n_threads=os.cpu_count(),
    n_gpu_layers=-1,
    logits_all=False,
    verbose=False
)

start = time.time()
out = llm("Hello, how are you?", max_tokens=1)
print("TTFT:", time.time() - start)
