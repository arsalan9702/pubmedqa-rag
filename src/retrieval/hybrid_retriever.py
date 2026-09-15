from __future__ import annotations

import yaml

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def normalize_scores(results: list[dict]) -> list[dict]:
    if not results:
        return results

    scores = [r["score"] for r in results]
    min_score, max_score = min(scores), max(scores)

    if max_score == min_score:
        for r in results:
            r["score"] = 1.0
        return results

    for r in results:
        r["score"] = (r["score"] - min_score) / (max_score - min_score)
    return results


class HybridRetriever:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or load_config()
        self.dense = DenseRetriever(self.config)
        self.bm25 = BM25Retriever(self.config)

        self.dense_weight = self.config["retrieval"]["dense_weight"]
        self.bm25_weight = self.config["retrieval"]["bm25_weight"]

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self.config["retrieval"]["top_k_hybrid"]

        dense_results = normalize_scores(self.dense.retrieve(query, top_k=k * 2))
        bm25_results = normalize_scores(self.bm25.retrieve(query, top_k=k * 2))

        combined: dict[str, dict] = {}

        for r in dense_results:
            combined[r["chunk_id"]] = {
                "chunk_id": r["chunk_id"],
                "pubid": r["pubid"],
                "text": r["text"],
                "score": r["score"] * self.dense_weight,
            }

        for r in bm25_results:
            if r["chunk_id"] in combined:
                combined[r["chunk_id"]]["score"] += r["score"] * self.bm25_weight
            else:
                combined[r["chunk_id"]] = {
                    "chunk_id": r["chunk_id"],
                    "pubid": r["pubid"],
                    "text": r["text"],
                    "score": r["score"] * self.bm25_weight,
                }

        ranked = sorted(combined.values(), key=lambda r: r["score"], reverse=True)
        return ranked[:k]


if __name__ == "__main__":
    retriever = HybridRetriever()
    results = retriever.retrieve("does exercise reduce heart disease risk", top_k=3)
    for r in results:
        print(f"{r['pubid']} (score={r['score']:.3f}) - {r['text'][:100]}")
