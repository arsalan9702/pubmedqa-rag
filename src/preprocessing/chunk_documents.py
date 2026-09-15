from __future__ import annotations

import json
import os

import yaml


def chunk_abstract(text: str, max_length: int = 256) -> list[str]:
    """
    split an abstract into word-bounded chunks

    text: the abstract text to chunk
    max_length: max number of words per chunk

    returns a list of text chunks
    """

    if text is None:
        raise TypeError("text must be a string, not None")
    if not text.strip():
        return []

    words = text.split()
    if len(words) <= max_length:
        return [text]

    chunks = []

    for i in range(0, len(words), max_length):
        chunk_words = words[i : i + max_length]
        chunks.append(" ".join(chunk_words))

    return chunks


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config()
    max_length = cfg.get("chunking", {}).get("max_length", 256)

    qa_pairs_file = cfg["data"]["qa_pairs_file"]
    chunks_file = cfg["data"]["chunks_file"]

    print(f"Reading records from file {qa_pairs_file}...")

    records = []
    with open(qa_pairs_file) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Loaded {len(records)} records. Chunking with max length {max_length}")

    chunk_records = []
    total_chunks = 0
    multi_chunk_count = 0

    for record in records:
        chunks = chunk_abstract(record["context"], max_length=max_length)
        total_chunks += len(chunks)
        if len(chunks) > 1:
            multi_chunk_count += 1

        for i, chunk_text in enumerate(chunks):
            chunk_records.append(
                {
                    "pubid": record["pubid"],
                    "chunk_id": f"{record['pubid']}_{i}",
                    "chunk_index": i,
                    "text": chunk_text,
                    "question": record["question"],
                    "final_decision": record.get("final_decision"),
                }
            )

    os.makedirs(os.path.dirname(chunks_file), exist_ok=True)
    with open(chunks_file, "w") as f:
        for chunk_record in chunk_records:
            f.write(json.dumps(chunk_record) + "\n")

    print(f"Saved {total_chunks} chunks to {chunks_file}")
    print(f"Abstracts split into multiple chunks: {multi_chunk_count} / {len(records)}")


if __name__ == "__main__":
    main()
