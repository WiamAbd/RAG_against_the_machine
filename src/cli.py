class RagCLI:
    """Expose the RAG pipeline commands."""

    def index(
        self,
        max_chunk_size: int = 2000
    ) -> None:
        """Build and save the index."""
        print("Source directory: data/raw")
        print("Index directory: data/processed")
        print(f"Maximum chunk size: {max_chunk_size}")

    def search(
        self,
        query: str,
        k: int = 5
    ) -> None:
        """Retrieve sources for one question."""
        print(f"Question: {query}")
        print(f"k: {k}")
        print("Index directory: data/processed")

    def search_dataset(
        self,
        dataset_path: str = (
            "data/datasets/UnansweredQuestions/dataset_docs_public.json"
        ),
        k: int = 5,
        save_directory: str = (
            "data/output/search_results/UnansweredQuestions"
        )
    ) -> None:
        """Retrieve sources for a question dataset."""
        print(f"Dataset: {dataset_path}")
        print(f"k: {k}")
        print(f"Output directory: {save_directory}")

    def answer(
        self,
        query: str,
        k: int = 5,
    ) -> None:
        """Retrieve sources and answer one question."""
        print(f"Question: {query}")
        print(f"k: {k}")

    def answer_dataset(
        self,
        student_search_results_path: str = (
            "data/output/search_results/UnansweredQuestions/"
            "dataset_docs_public.json"
        ),
        save_directory: str = (
            "data/output/search_results_and_answer/UnansweredQuestions"
        ),
    ) -> None:
        """Generate answers from saved retrieval results."""
        print(f"Search results: {student_search_results_path}")
        print(f"Output directory: {save_directory}")

    def evaluate(
        self,
        student_search_results_path: str = (
            "data/output/search_results/UnansweredQuestions/"
            "dataset_docs_public.json"
        ),
        dataset_path: str = (
            "data/datasets/AnsweredQuestions/dataset_docs_public.json"
        ),
    ) -> None:
        """Evaluate retrieval against reference sources."""
        print(f"Search results: {student_search_results_path}")
        print(f"Reference dataset: {dataset_path}")