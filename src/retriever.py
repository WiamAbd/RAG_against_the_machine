"""Retrieve repository chunks using BM25."""

from heapq import nlargest

from rank_bm25 import BM25Okapi

from .indexer import tokenize
from .models import Chunk
from tqdm import tqdm

from .models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


def retrieve(
    query: str,
    index: BM25Okapi,
    chunks: list[Chunk],
    k: int = 5,
) -> list[Chunk]:
    """Return up to k chunks ranked by their BM25 scores."""
    if not isinstance(query, str):
        raise ValueError("query must be a string.")

    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer.")

    query_tokens = tokenize(query)

    if not query_tokens:
        raise ValueError("The query must contain searchable terms.")

    if index.corpus_size != len(chunks):
        raise ValueError("The index and chunks have different sizes.")

    scores = index.get_scores(query_tokens)

    if not scores.any():
        return []

    best_positions = nlargest(
        k,
        range(len(chunks)),
        key=lambda position: scores[position],
    )

    return [chunks[position] for position in best_positions]


def retrieve_dataset(
    dataset: RagDataset,
    index: BM25Okapi,
    chunks: list[Chunk],
    k: int = 5,
) -> StudentSearchResults:
    """Retrieve sources for every question using an already-loaded index."""
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer.")

    search_results: list[MinimalSearchResults] = []

    for question in tqdm(
        dataset.rag_questions,
        desc="Retrieving sources",
    ):
        retrieved_chunks = retrieve(
            query=question.question,
            index=index,
            chunks=chunks,
            k=k,
        )

        sources = [
            MinimalSource(
                file_path=chunk.file_path,
                first_character_index=chunk.first_character_index,
                last_character_index=chunk.last_character_index,
            )
            for chunk in retrieved_chunks
        ]

        search_results.append(
            MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=sources,
            )
        )

    return StudentSearchResults(
        search_results=search_results,
        k=k,
    )
