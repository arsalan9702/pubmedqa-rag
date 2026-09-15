"""Build Croma vector index from chunked PubMed extracts"""

import json

import chromadb
import yaml
from sentence_transformers import SentenceTransformer


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_chunks(chunks_file: str) -> list[dict]:
    chunks = []
    with open(chunks_file) as f:
        for line in f:
            chunks.append(json.loads(line))

    return chunks


def main() -> None:
    cfg = load_config()

    chunks_file = cfg["data"]["chunks_file"]
    embedding_model_name = cfg["embedding"]["model_name"]
    persist_dir = cfg["vector_store"]["persist_dir"]
    collection_name = cfg["vector_store"]["collection_name"]
    batch_size = cfg["embedding"]["batch_size"]

    print(f"Loading chunks from {chunks_file}...")
    chunks = load_chunks(chunks_file)
    print(f"Loaded {len(chunks)} chunks")

    print(f"Loading embeddings model {embedding_model_name}...")
    model = SentenceTransformer(embedding_model_name)

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]

    metadatas = [
        {
            "pubid": c["pubid"],
            "chunk_index": c["chunk_index"],
            "question": c["question"],
            "final_decision": c["final_decision"] or "",
        }
        for c in chunks
    ]

    print("Encoding chunks...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).tolist()

    print(f"Writing to Chroma at {persist_dir}...")
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name=collection_name)

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"Indexed {collection.count()} chunks into collection '{collection_name}'")


if __name__ == "__main__":
    main()
