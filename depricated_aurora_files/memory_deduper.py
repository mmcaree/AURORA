from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import json

# Load summaries
with open("long_term_memory.json", "r", encoding="utf-8") as f:
    summaries = json.load(f)

# Use your embedder
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(summaries)

# Run KMeans
n_clusters = min(10, len(summaries))  # 10 or less clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(embeddings)

# Pick representative (closest to each centroid)
from sklearn.metrics import pairwise_distances_argmin_min
closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
distinct_summaries = [summaries[i] for i in closest]

# Save filtered version
with open("long_term_memory_filtered.json", "w", encoding="utf-8") as f:
    json.dump(distinct_summaries, f, indent=2)
