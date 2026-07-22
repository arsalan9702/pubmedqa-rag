from __future__ import annotations


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
