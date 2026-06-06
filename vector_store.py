"""
vector_store.py
===============
PURPOSE: Convert resume text → vectors → store in FAISS for fast search.

KEY LIBRARIES:
  - SentenceTransformer : converts text into numerical vectors (embeddings)
  - FAISS               : stores vectors and finds similar ones fast
  - LangChain TextSplitter : splits long text into smaller chunks

INTERVIEW EXPLANATION:
"Computers can't understand words directly. SentenceTransformer converts
text into a list of numbers called a 'vector'. Similar sentences get
similar vectors. FAISS then lets us search: which resume vector is
closest to the job description vector? That's our similarity match."

FLOW:
  Resume Text
    → split into chunks (500 chars each)
    → SentenceTransformer encodes each chunk into a 384-dim vector
    → vectors stored in FAISS IndexFlatL2
    → job description also encoded
    → FAISS search returns closest resume chunks
"""

import os
from dotenv import load_dotenv
load_dotenv()  # Load .env before anything that needs env vars

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Load the embedding model once (not inside a function, so it loads only once)
# "all-MiniLM-L6-v2" is small, fast, and good for semantic similarity tasks
# It produces 384-dimensional vectors
print("[vector_store] Loading SentenceTransformer model...")
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("[vector_store] Model loaded!")


def split_text_into_chunks(text: str, candidate_name: str) -> list:
    """
    Splits long resume text into smaller overlapping chunks.

    WHY CHUNK?
    - Embedding models work best on short, focused text
    - 500 chars ≈ 2-3 sentences — enough context per chunk
    - 50 char overlap ensures we don't cut mid-sentence

    Returns:
        List of dicts: [{"text": "...", "candidate": "John Doe"}, ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    return [{"text": chunk, "candidate": candidate_name} for chunk in chunks]


def build_faiss_index(resumes: dict) -> tuple:
    """
    Builds a FAISS vector index from all resume texts.

    Args:
        resumes: {"John Doe": "full resume text...", ...}

    Returns:
        (faiss_index, all_chunks_list)
    """
    all_chunks = []

    # Step 1: Chunk all resumes
    for candidate_name, resume_text in resumes.items():
        chunks = split_text_into_chunks(resume_text, candidate_name)
        all_chunks.extend(chunks)
        print(f"[vector_store] {candidate_name}: {len(chunks)} chunks")

    # Step 2: Extract just text for encoding
    texts = [chunk["text"] for chunk in all_chunks]

    # Step 3: Encode with SentenceTransformer
    # encode() returns a numpy array of shape (num_texts, 384)
    print(f"[vector_store] Encoding {len(texts)} chunks...")
    vectors = EMBEDDING_MODEL.encode(texts, show_progress_bar=False)

    # Step 4: Convert to float32 (FAISS requirement)
    vectors_np = np.array(vectors, dtype=np.float32)

    # Step 5: Build FAISS index
    # IndexFlatL2 = exact search using Euclidean (L2) distance
    # Smaller distance = more similar
    dimension = vectors_np.shape[1]  # 384 for all-MiniLM-L6-v2
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors_np)

    print(f"[vector_store] FAISS index built: {index.ntotal} vectors, dim={dimension}")
    return index, all_chunks


def search_similar(index, all_chunks: list, query_text: str, top_k: int = 3) -> dict:
    """
    Finds the most similar resume chunks to the job description.

    Args:
        index     : FAISS index
        all_chunks: List of chunk dicts with 'text' and 'candidate'
        query_text: Job description text
        top_k     : How many results per candidate

    Returns:
        {"John Doe": ["relevant chunk 1", ...], ...}
    """
    # Encode the job description into a vector
    query_vector = EMBEDDING_MODEL.encode([query_text])
    query_np = np.array(query_vector, dtype=np.float32)

    # Search FAISS — D=distances, I=indices of top matches
    num_candidates = len(set(c["candidate"] for c in all_chunks))
    search_k = min(top_k * num_candidates, len(all_chunks))
    distances, indices = index.search(query_np, search_k)

    # Group results by candidate
    results = {}
    for idx in indices[0]:
        if 0 <= idx < len(all_chunks):
            chunk = all_chunks[idx]
            name = chunk["candidate"]
            if name not in results:
                results[name] = []
            if len(results[name]) < top_k:
                results[name].append(chunk["text"])

    return results
