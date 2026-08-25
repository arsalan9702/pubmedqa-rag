from __future__ import annotations

import json
import re

import yaml
from rank_bm25 import BM25Okapi


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_chunks(chunks_file: str) -> list[dict]:
    chunks = []
    with open(chunks_file) as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or load_config()
        chunks_file = self.config["data"]["chunks_file"]

        self.chunks = load_chunks(chunks_file)
        tokenized_corpus = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self.config["retrieval"]["top_k_bm25"]
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        retrieved = []
        for i in ranked_indices:
            chunk = self.chunks[i]
            retrieved.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "pubid": chunk["pubid"],
                    "text": chunk["text"],
                    "score": float(scores[i]),
                }
            )
        return retrieved


if __name__ == "__main__":
    retriever = BM25Retriever()
    results = retriever.retrieve("does exercise reduce heart disease risk", top_k=3)
    for r in results:
        print(f"{r['pubid']} (score={r['score']:.3f}) - {r['text'][:100]}")
