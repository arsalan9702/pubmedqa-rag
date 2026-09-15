import pytest

from src.preprocessing.chunk_documents import chunk_abstract


def test_chunk_short_abstract_returns_single_chunk():
    text = "This is a short abstract about a clinical trial."
    chunks = chunk_abstract(text, max_length=256)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_empty_string_returns_empty_list():
    chunks = chunk_abstract("", max_length=256)
    assert chunks == []


def test_chunk_respects_max_length():
    text = "word " * 500  # long enough to require splitting
    chunks = chunk_abstract(text, max_length=100)
    assert all(len(c.split()) <= 100 for c in chunks)


def test_chunk_none_input_raises():
    with pytest.raises(TypeError):
        chunk_abstract(None, max_length=256)
