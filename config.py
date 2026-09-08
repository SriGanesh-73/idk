"""
config.py — Central configuration for the RAG pipeline.

All tunable parameters live here so you can adjust the pipeline
without digging through multiple files.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────
# Directory where you place your college PDF documents.
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Directory where ChromaDB stores its persistent database.
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# ─── ChromaDB ────────────────────────────────────────────────────────
# Name of the ChromaDB collection that holds all document chunks.
COLLECTION_NAME = "college_knowledge"

# Distance metric for vector similarity search in ChromaDB.
# 'cosine' calculates cosine distance (1 - cosine_similarity).
DISTANCE_METRIC = "cosine"

# ─── Embedding Model ────────────────────────────────────────────────
# ChromaDB's built-in DefaultEmbeddingFunction uses all-MiniLM-L6-v2
# via ONNX runtime. No separate configuration is needed — if you want
# a different model, swap the embedding function in embeddings.py.

# ─── Chunking ───────────────────────────────────────────────────────
# Maximum number of characters per chunk.
CHUNK_SIZE = 500

# Number of overlapping characters between consecutive chunks.
# Overlap helps preserve context that falls on chunk boundaries.
CHUNK_OVERLAP = 100

# ─── Retrieval ──────────────────────────────────────────────────────
# Number of most-relevant chunks to retrieve for each question.
TOP_K = 5

# ─── LLM Models & Fallback Hierarchy ───────────────────────────────
# Primary model followed by fallback models in order of priority.
# If a model encounters 503 high demand, 429 rate limit, or 404 deprecation,
# the system automatically fails over to the next available model.
LLM_MODELS = [
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

