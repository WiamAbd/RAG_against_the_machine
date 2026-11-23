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

    tokenized_chunks = [
        tokenize(chunk.text)
        for chunk in chunks
    ]

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
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_index(
    index_directory: str = "data/processed",
) -> tuple[BM25Okapi, list[Chunk]]:
    """Load the BM25 index and its ordered chunks."""
    index_path = Path(index_directory) / "bm25.pkl"

    with index_path.open("rb") as file:
        data = pickle.load(file)

    return data["index"], data["chunks"]

if __name__ == "__main__":
    texts = [
        "The KV cache stores attention keys and values.",
        "Install the package using pip.",
        "The server accepts HTTP requests.",
    ]

    chunks = [
        Chunk(
            file_path=f"example_{i}.md",
            first_character_index=0,
            last_character_index=len(text),
            text=text,
        )
        for i, text in enumerate(texts)
    ]

    index = build_index(chunks)

    query = "attention cache"
    scores = index.get_scores(tokenize(query))

    for chunk, score in zip(chunks, scores):
        print(f"{chunk.file_path}: {score:.4f}")

    best_index = int(scores.argmax())

    print(f"\nBest match: {chunks[best_index].text}")

    assert len(scores) == len(chunks)
    assert best_index == 0
    assert scores[0] > 0

    print("BM25 checks passed.")
    save_index(index, chunks)

    index_path = Path("data/processed/bm25.pkl")

    assert index_path.is_file()
    assert index_path.stat().st_size > 0

    print(f"Index saved to {index_path}")
    loaded_index, loaded_chunks = load_index()

    loaded_scores = loaded_index.get_scores(tokenize(query))

    assert loaded_chunks == chunks
    assert (loaded_scores == scores).all()

    print("Loading check passed: chunks and scores are unchanged.")