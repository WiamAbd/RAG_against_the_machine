from time import perf_counter

from .read_files import load_documents
from .chunking import chunk_documents
from .indexer import build_index, save_index, load_index
from .retriever import retrieve, retrieve_dataset
from .datasets import (
    load_dataset,
    load_search_results,
    save_search_results,
)
from .evaluation import dataset_recall
from .models import UnansweredQuestion
from .datasets import save_answers
from .pipeline import answer_question, generate_dataset_answers


class RagCLI:
    """Expose the RAG pipeline commands."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Read repository files, chunk them, and save a BM25 index."""
        if type(max_chunk_size) is not int:
            raise ValueError("max_chunk_size must be an integer.")

        if not 1 <= max_chunk_size <= 2000:
            raise ValueError(
                "max_chunk_size must be between 1 and 2000."
            )

        start = perf_counter()

        documents = load_documents("data/raw")
        print(f"Loaded {len(documents)} documents.")

        chunks = chunk_documents(
            documents,
            max_chunk_size=max_chunk_size,
            min_chunk_size=min(200, max_chunk_size),
        )
        print(f"Created {len(chunks)} chunks.")

        bm25_index = build_index(chunks)

        save_index(bm25_index, chunks, "data/processed")
        saving_end = perf_counter()

        print(f"Total indexing time: {(saving_end - start)/60:.2f} minutes")
        print("Index saved to data/processed/bm25.pkl")

    def search(self, query: str, k: int = 5) -> None:
        """Retrieve and display the best matching repository chunks."""
        index, chunks = load_index("data/processed")

        results = retrieve(
            query=query,
            index=index,
            chunks=chunks,
            k=k,
        )

        if not results:
            print("No matching chunks found.")
            return

        for number, chunk in enumerate(results, start=1):
            print(f"\nResult {number}: {chunk.file_path}")
            print(
                f"Characters: {chunk.first_character_index}"
                f"–{chunk.last_character_index}"
            )
            print(chunk.text)

    def search_dataset(
        self,
        dataset_path: str = (
            "datasets_public/public/UnansweredQuestions/"
            "dataset_docs_public.json"
        ),
        k: int = 5,
        save_directory: str = (
            "data/output/search_results/UnansweredQuestions"
        ),
    ) -> None:
        """Retrieve sources for a question dataset and save them as JSON."""
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer.")

        start = perf_counter()

        dataset = load_dataset(dataset_path)
        index, chunks = load_index("data/processed")

        results = retrieve_dataset(
            dataset=dataset,
            index=index,
            chunks=chunks,
            k=k,
        )

        output_path = save_search_results(
            results=results,
            dataset_path=dataset_path,
            save_directory=save_directory,
        )

        print(f"\nProcessed {len(results.search_results)} questions.")
        print(f"Results saved to {output_path}")
        print(f"Total: {perf_counter() - start:.2f} seconds")

    def answer(self, query: str, k: int = 5) -> None:
        """Retrieve sources and generate a structured answer."""
        # Import here so other CLI commands do not import the LLM module.
        from .llm import Small_LLM_Model

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string.")

        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer.")

        index, chunks = load_index("data/processed")

        retrieved_chunks = retrieve(
            query=query,
            index=index,
            chunks=chunks,
            k=k,
        )

        question = UnansweredQuestion(question=query)
        llm = Small_LLM_Model()

        result = answer_question(
            question=question,
            chunks=retrieved_chunks,
            llm=llm,
        )

        print(result.model_dump_json(indent=2))

    def answer_dataset(
        self,
        student_search_results_path: str = (
            "data/output/search_results/UnansweredQuestions/"
            "dataset_docs_public.json"
        ),
        save_directory: str = (
            "data/output/search_results_and_answer/"
            "UnansweredQuestions"
        ),
    ) -> None:
        """Generate and save answers from previously retrieved sources."""
        from .llm import Small_LLM_Model

        start = perf_counter()

        results = load_search_results(student_search_results_path)
        print(f"Loaded {len(results.search_results)} questions.")

        llm = Small_LLM_Model()

        answers = generate_dataset_answers(
            results=results,
            llm=llm,
        )

        output_path = save_answers(
            results=answers,
            student_search_results_path=student_search_results_path,
            save_directory=save_directory,
        )

        print(f"\nGenerated {len(answers.search_results)} answers.")
        print(f"Answers saved to {output_path}")
        print(f"Total: {perf_counter() - start:.2f} seconds")

    def evaluate(
        self,
        student_search_results_path: str = (
            "data/output/search_results/UnansweredQuestions/"
            "dataset_docs_public.json"
        ),
        dataset_path: str = (
            "datasets_public/public/AnsweredQuestions/"
            "dataset_docs_public.json"
        ),
    ) -> None:
        """Evaluate saved retrieval results against reference sources."""
        results = load_search_results(student_search_results_path)
        reference_dataset = load_dataset(dataset_path)

        print(f"Search results: {student_search_results_path}")
        print(f"Reference dataset: {dataset_path}")
        print(f"Reference questions: {len(reference_dataset.rag_questions)}")

        for k in (1, 3, 5, 10):
            if k <= results.k:
                score = dataset_recall(
                    results=results,
                    reference_dataset=reference_dataset,
                    k=k,
                )

                print(f"Recall@{k}: {score:.3f} ({score:.1%})")
