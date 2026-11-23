"""Evaluate retrieved source locations against reference sources."""

from .models import MinimalSource
from .models import AnsweredQuestion, RagDataset, StudentSearchResults

def source_iou(
    retrieved: MinimalSource,
    reference: MinimalSource,
) -> float:
    """Compute character-range IoU for sources in the same file."""
    if retrieved.file_path != reference.file_path:
        return 0.0

    intersection = max(
        0,
        min(
            retrieved.last_character_index,
            reference.last_character_index,
        )
        - max(
            retrieved.first_character_index,
            reference.first_character_index,
        ),
    )

    retrieved_length = (
        retrieved.last_character_index
        - retrieved.first_character_index
    )
    reference_length = (
        reference.last_character_index
        - reference.first_character_index
    )

    union = retrieved_length + reference_length - intersection

    if union <= 0:
        return 0.0

    return intersection / union

def source_matches(
    retrieved: MinimalSource,
    reference: MinimalSource,
    threshold: float = 0.05,
) -> bool:
    """Check whether source paths match and their IoU reaches the threshold."""
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be greater than 0 and at most 1.")

    return source_iou(retrieved, reference) >= threshold
def question_recall(
    retrieved_sources: list[MinimalSource],
    reference_sources: list[MinimalSource],
    k: int = 5,
) -> float:
    """Calculate recall@k for one question."""
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer.")

    if not reference_sources:
        raise ValueError("The question has no reference sources.")

    top_sources = retrieved_sources[:k]
    found = 0

    for reference in reference_sources:
        if any(
            source_matches(retrieved, reference)
            for retrieved in top_sources
        ):
            found += 1

    return found / len(reference_sources)

def dataset_recall(
    results: StudentSearchResults,
    reference_dataset: RagDataset,
    k: int = 5,
) -> float:
    """Calculate mean recall@k across all reference questions."""
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer.")

    if k > results.k:
        raise ValueError("k exceeds the saved retrieval depth.")

    if not reference_dataset.rag_questions:
        raise ValueError("The reference dataset is empty.")

    results_by_id = {}

    for result in results.search_results:
        if result.question_id in results_by_id:
            raise ValueError(f"Duplicate result ID: {result.question_id}")

        results_by_id[result.question_id] = result

    total_recall = 0.0
    reference_ids = set()

    for question in reference_dataset.rag_questions:
        if not isinstance(question, AnsweredQuestion):
            raise ValueError("Reference questions must contain sources and answers.")

        if question.question_id in reference_ids:
            raise ValueError(f"Duplicate reference ID: {question.question_id}")

        reference_ids.add(question.question_id)

        result = results_by_id.get(question.question_id)

        # Missing results count as zero recall.
        retrieved_sources = (
            result.retrieved_sources if result is not None else []
        )

        total_recall += question_recall(
            retrieved_sources=retrieved_sources,
            reference_sources=question.sources,
            k=k,
        )

    return total_recall / len(reference_dataset.rag_questions)

if __name__ == "__main__":
    from .datasets import load_dataset, load_search_results

    # Test character-range overlap.
    retrieved = MinimalSource(
        file_path="example.py",
        first_character_index=0,
        last_character_index=100,
    )

    reference = MinimalSource(
        file_path="example.py",
        first_character_index=50,
        last_character_index=150,
    )

    other_file = reference.model_copy(
        update={"file_path": "other.py"}
    )

    non_overlapping = MinimalSource(
        file_path="example.py",
        first_character_index=200,
        last_character_index=300,
    )

    assert abs(source_iou(retrieved, reference) - 1 / 3) < 1e-9
    assert source_iou(retrieved, retrieved) == 1.0
    assert source_iou(retrieved, other_file) == 0.0
    assert source_iou(retrieved, non_overlapping) == 0.0

    print("IoU checks passed.")

    # Test source matching.
    assert source_matches(retrieved, reference) is True
    assert source_matches(retrieved, other_file) is False
    assert source_matches(retrieved, non_overlapping) is False

    print("Source matching checks passed.")

    # Test recall for one question.
    reference_sources = [reference, non_overlapping]
    retrieved_sources = [retrieved, non_overlapping]

    assert question_recall(
        retrieved_sources, reference_sources, k=1
    ) == 0.5

    assert question_recall(
        retrieved_sources, reference_sources, k=2
    ) == 1.0

    assert question_recall(
        [], reference_sources, k=5
    ) == 0.0

    print("Question recall checks passed.")

    # Load saved retrieval results and reference answers.
    results = load_search_results(
        "data/output/search_results/UnansweredQuestions/"
        "dataset_code_public.json"
    )

    reference_dataset = load_dataset(
        "datasets_public/public/AnsweredQuestions/"
        "dataset_code_public.json"
    )

    print(f"\nLoaded results for {len(results.search_results)} questions.")
    print(f"Loaded {len(reference_dataset.rag_questions)} reference questions.")
    print(f"Saved retrieval depth: {results.k}")

    # Measure retrieval quality on the real dataset.
    score = dataset_recall(
        results=results,
        reference_dataset=reference_dataset,
        k=5,
    )

    print(f"\nDocs recall@5: {score:.3f} ({score:.1%})")