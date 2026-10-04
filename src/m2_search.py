from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    if not text.strip():
        return ""
    import unicodedata
    from underthesea import word_tokenize

    segmented = word_tokenize(unicodedata.normalize("NFC", text), format="text")
    return segmented.replace("_", " ")


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        import re
        from rank_bm25 import BM25Okapi

        self.documents = [c for c in chunks if c["text"].strip()]
        self.corpus_tokens = [re.findall(r"\w+", segment_vietnamese(c["text"]).lower())
                              for c in self.documents]
        self.bm25 = BM25Okapi(self.corpus_tokens) if any(self.corpus_tokens) else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or top_k <= 0 or not query.strip():
            return []
        import re

        tokens = re.findall(r"\w+", segment_vietnamese(query).lower())
        scores = self.bm25.get_scores(tokens)
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [SearchResult(text=self.documents[i]["text"], score=float(scores[i]),
                             metadata=dict(self.documents[i].get("metadata", {})), method="bm25")
                for i in indices if scores[i] > 0]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
        except Exception:
            self.client = QdrantClient(":memory:")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, PointStruct, VectorParams

        chunks = [c for c in chunks if c["text"].strip()]
        vectors = None
        dimension = EMBEDDING_DIM
        if chunks:
            vectors = self._get_encoder().encode(
                [c["text"] for c in chunks], batch_size=16,
                show_progress_bar=True, normalize_embeddings=True,
            )
            dimension = len(vectors[0])
        if self.client.collection_exists(collection_name=collection):
            self.client.delete_collection(collection_name=collection)
        self.client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
        if vectors is not None:
            for start in range(0, len(chunks), 64):
                points = [PointStruct(
                    id=i, vector=vectors[i].tolist(),
                    payload={**chunks[i].get("metadata", {}), "text": chunks[i]["text"]},
                ) for i in range(start, min(start + 64, len(chunks)))]
                self.client.upsert(collection_name=collection, points=points, wait=True)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if top_k <= 0 or not query.strip() or not self.client.collection_exists(collection_name=collection):
            return []
        if self.client.count(collection_name=collection, exact=True).count == 0:
            return []
        query_vector = self._get_encoder().encode(query, normalize_embeddings=True).tolist()
        response = self.client.query_points(
            collection_name=collection, query=query_vector, limit=top_k, with_payload=True,
        )
        results = []
        for point in response.points:
            metadata = dict(point.payload or {})
            text = metadata.pop("text", "")
            results.append(SearchResult(text=text, score=float(point.score), metadata=metadata, method="dense"))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if top_k <= 0:
        return []
    scores, documents = {}, {}
    for ranked_list in results_list:
        seen = set()
        for rank, result in enumerate(ranked_list):
            identity = (result.metadata.get("source", ""), result.metadata.get("chunk_id", result.text))
            if identity in seen:
                continue
            seen.add(identity)
            documents.setdefault(identity, result)
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (k + rank + 1)
    identities = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [SearchResult(text=documents[key].text, score=scores[key],
                         metadata=dict(documents[key].metadata), method="hybrid")
            for key in identities]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
