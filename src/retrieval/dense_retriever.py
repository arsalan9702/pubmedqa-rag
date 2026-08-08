from __future__ import annotations

import chromadb
import yaml
from sentence_transformers import SentenceTransformer

def load_config(path:str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

class DenseRetriever:
    """ Semantic retriever backed by chromadb store"""

    def __init__(self, config: dict | None = None)->None:
        self.config = config or load_config()

        model_name = self.config["embedding"]["model_name"]
        persist_dir = self.config["vector_store"]["persist_dir"]
        collection_name = self.config["vector_store"]["collection_name"]

        self.model = SentenceTransformer(model_name)
        client = chromadb.PersistentClient(path=persist_dir)
        self.collection = client.get_collection(name=collection_name)

    def retrieve(self, query:str, top_k:int | None=None) -> list[dict]:
        k = top_k or self.config["retrieval"]["top_k_dense"]
        query_embedding = self.model.encode(query, convert_to_numpy=True).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        retrieved=[]
        for chunk_id, doc, meta, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            retrieved.append(
                {
                    "chunk_id": chunk_id,
                    "pubid": meta["pubid"],
                    "text": doc,
                    "score": 1-distance #converting distance to similarity
                }
            )
        return retrieved

if __name__=="__main__":
    retriever = DenseRetriever()
    results = retriever.retrieve("does exercise reduce heart disease risk", top_k=3)
    for r in results:
        print(f"{r['pubid']} (score={r['score']:.3f}) - {r['text'][:100]}")