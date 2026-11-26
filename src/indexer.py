"""Build and persist a BM25 index for repository chunks."""

import re
from rank_bm25 import BM25Okapi
from .models import Chunk
import pickle
from pathlib import Path


def tokenize(text: str) -> list[str]:
    """Lowercase text and extract terms, splitting at underscores."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_index(chunks: list[Chunk]) -> BM25Okapi:
    """Build a BM25 index from chunk texts, preserving their order."""
    if not chunks:
        raise ValueError("Cannot build an index without chunks.")

    tokenized_chunks = []

    for chunk in chunks:
        context = " ".join(
            chunk.class_names + chunk.function_names
        )

        searchable_text = f"{context}\n{chunk.text}"

        tokenized_chunks.append(tokenize(searchable_text))

    if not any(tokenized_chunks):
        raise ValueError("The chunks contain no searchable terms.")

    return BM25Okapi(tokenized_chunks, k1=1.5, b=0.75)


def save_index(
    index: BM25Okapi,
    chunks: list[Chunk],
    index_directory: str = "data/processed",
) -> None:
    """Save the BM25 index and its ordered chunks together."""
    if index.corpus_size != len(chunks):
        raise ValueError(
            "The index and chunks must contain the same number of documents."
        )

    directory = Path(index_directory)
    directory.mkdir(parents=True, exist_ok=True)

    index_path = directory / "bm25.pkl"

    data = {
        "index": index,
        "chunks": chunks,
    }

    with index_path.open("wb") as file:
        pickle.dump(data, file)


def load_index(
    index_directory: str = "data/processed",
) -> tuple[BM25Okapi, list[Chunk]]:
    """Load the BM25 index and its ordered chunks."""
    index_path = Path(index_directory) / "bm25.pkl"

    with index_path.open("rb") as file:
        data = pickle.load(file)

    return data["index"], data["chunks"]
