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

if __name__ == "__main__":
    from time import perf_counter

    from .datasets import load_dataset
    from .indexer import load_index

    dataset = load_dataset(
        "datasets_public/public/UnansweredQuestions/"
        "dataset_docs_public.json"
    )

    start = perf_counter()

    index, chunks = load_index()

    results = retrieve_dataset(
        dataset=dataset,
        index=index,
        chunks=chunks,
        k=5,
    )

    elapsed = perf_counter() - start

    print(f"\nProcessed {len(results.search_results)} questions.")
    print(f"Index loading and retrieval: {elapsed:.2f} seconds")

    for result in results.search_results[:3]:
        print(f"\nQuestion: {result.question}")

        for source in result.retrieved_sources:
            print(
                source.file_path,
                source.first_character_index,
                source.last_character_index,
            )

    assert len(results.search_results) == len(dataset.rag_questions)

    assert [
        result.question_id for result in results.search_results
    ] == [
        question.question_id for question in dataset.rag_questions
    ]

    assert all(
        len(result.retrieved_sources) <= results.k
        for result in results.search_results
    )

    print("\nBatch retrieval checks passed.")