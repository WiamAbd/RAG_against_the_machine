"""Read question datasets."""

from pathlib import Path

from .models import (
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)


def load_dataset(dataset_path: str) -> RagDataset:
    """Read a JSON dataset and validate its structure."""
    path = Path(dataset_path)

    text = path.read_text(encoding="utf-8")

    return RagDataset.model_validate_json(text)


def save_search_results(
    results: StudentSearchResults,
    dataset_path: str,
    save_directory: str,
) -> Path:
    """Save retrieval results using the input dataset's filename."""
    directory = Path(save_directory)
    directory.mkdir(parents=True, exist_ok=True)

    output_path = directory / Path(dataset_path).name

    output_path.write_text(
        results.model_dump_json(indent=2),
        encoding="utf-8",
    )

    return output_path


def load_search_results(
    results_path: str,
) -> StudentSearchResults:
    """Read and validate saved retrieval results."""
    text = Path(results_path).read_text(encoding="utf-8")

    return StudentSearchResults.model_validate_json(text)


def save_answers(
    results: StudentSearchResultsAndAnswer,
    student_search_results_path: str,
    save_directory: str,
) -> Path:
    """Save generated answers using the search results' filename."""
    directory = Path(save_directory)
    directory.mkdir(parents=True, exist_ok=True)

    output_path = directory / Path(student_search_results_path).name

    output_path.write_text(
        results.model_dump_json(indent=2),
        encoding="utf-8",
    )

    return output_path


if __name__ == "__main__":
    dataset = load_dataset(
        "datasets_public/public/UnansweredQuestions/dataset_docs_public.json"
    )

    print(f"Loaded {len(dataset.rag_questions)} questions.")

    for question in dataset.rag_questions[:3]:
        print(f"\nID: {question.question_id}")
        print(f"Question: {question.question}")
