"""Retrieval evaluation metrics: recall@k and MRR."""

from __future__ import annotations

import json

def load_qa_pairs(qa_pairs_file: str) -> list[dict]:
   records = []
   with open(qa_pairs_file) as f:
      for line in f:
         records.append(json.loads(line))
   return records

def recall_at_k(retrieved_pubids: list[int], gold_pubid: int) -> int:
   # checks whether the gold document appears anywhere in the retrieved set
   return 1 if gold_pubid in retrieved_pubids else 0

def reciprocal_rank(retrieved_pubids: list[int], gold_pubid: int) -> float:
   # compute the reciprocal rank of the gold document in the retrieved list
   for i, pubid in enumerate(retrieved_pubids):
      if pubid == gold_pubid:
         return 1.0/(i+1)
         
   return 0.0

def evaluate_retriever(retriever, qa_pairs: list[dict], top_k: int) -> dict:
   recalls=[]
   reciprocal_ranks=[]

   for record in qa_pairs:
      query = record["question"]
      gold_pubid = record["pubid"]

      results = retriever.retrieve(query, top_k=top_k)
      retrieved_pubids = [r["pubid"] for r in results]

      recalls.append(recall_at_k(retrieved_pubids, gold_pubid))
      reciprocal_ranks.append(reciprocal_rank(retrieved_pubids, gold_pubid))

   return{
      "recall_at_k": sum(recalls)/len(recalls),
      "mrr": sum(reciprocal_ranks)/len(reciprocal_ranks),
      "n_queries": len(qa_pairs),
   }

def main() -> None:
   import yaml

   from src.retrieval.bm25_retriever import BM25Retriever
   from src.retrieval.dense_retriever import DenseRetriever
   from src.retrieval.hybrid_retriever import HybridRetriever

   with open("configs/config.yaml") as f:
      config = yaml.safe_load(f)

   qa_pairs = load_qa_pairs(config["data"]["qa_pairs_file"])
   top_k = config["retrieval"]["top_k_hybrid"]

   print(f"Evaluating retrievers on {len(qa_pairs)} questions, top_k = {top_k}...\n")

   retrievers = {
      "Dense": DenseRetriever(config),
      "BM25": BM25Retriever(config),
      "Hybrid": HybridRetriever(config)
   }

   for name, retriever in retrievers.items():
      print(f"Running {name}...")
      metrics = evaluate_retriever(retriever, qa_pairs, top_k)
      print(f"Recall@{top_k}: {metrics["recall_at_k"]:.4f}")
      print(f"MRR {metrics['mrr']:.4f}\n")

if __name__ == "__main__":
   main()
   

