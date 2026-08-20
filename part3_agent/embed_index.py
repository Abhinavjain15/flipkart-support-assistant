"""
Embeds all policy chunks with a free local sentence-transformer and builds
a Faiss inner-product index (on L2-normalized vectors, equivalent to cosine similarity).
No API key, no account, fully offline after the model is first downloaded.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from knowledge_base.policies import build_all_chunks

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None
_index = None
_chunk_meta = (
    None  # list of (chunk_id, chunk_text, parent_doc_id), aligned to index rows
)


def get_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def build_index():
    global _index, _chunk_meta
    all_chunks, _ = build_all_chunks()
    texts = [text for _, text, _ in all_chunks]

    embedder = get_embedder()
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)  # normalize so inner product == cosine similarity

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    _index = index
    _chunk_meta = all_chunks
    return index, all_chunks


def search(query: str, k: int = 3):
    """Returns list of dicts: chunk_id, text, parent_doc, score (cosine similarity)."""
    if _index is None:
        build_index()

    embedder = get_embedder()
    query_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)

    scores, indices = _index.search(query_vec, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk_id, text, doc_id = _chunk_meta[idx]
        results.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "parent_doc": doc_id,
                "score": float(score),
            }
        )
    return results


if __name__ == "__main__":
    build_index()
    print(f"Indexed {len(_chunk_meta)} chunks with {MODEL_NAME}\n")

    test_query = "How long do I have to return a shirt I bought?"
    results = search(test_query, k=3)
    print(f"Query: {test_query}\n")
    for r in results:
        print(f"  score={r['score']:.4f}  doc={r['parent_doc']:30s}  text={r['text']}")
