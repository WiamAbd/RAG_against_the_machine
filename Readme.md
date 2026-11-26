*This project has been created as part of the 42 curriculum by wabdella.*

# RAG against the machine

## Description

A Retrieval-Augmented Generation (RAG) system that answers questions about the provided **vLLM 0.10.1** codebase. It retrieves source excerpts using **BM25**, generates answers with **Qwen/Qwen3-0.6B**, and measures retrieval quality using recall@k.

Built with Python, uv, Python Fire, Pydantic, tqdm, rank-bm25, PyTorch, and Transformers. The model is used locally for inference, without fine-tuning.

## Instructions

### Installation and data

Use Python 3.11+ and uv. Run all commands from the repository root:

```bash
uv sync
```

Extract the supplied corpus under `data/raw/vllm-0.10.1/`, without an extra nested repository directory. Public datasets in the examples use `datasets_public/public/`, containing `AnsweredQuestions/` and `UnansweredQuestions/`. Other dataset locations can be passed through CLI arguments.

The first answering command downloads Qwen unless it is already cached. CPU inference works; a compatible GPU accelerates generation. On Windows, run consistently inside WSL and do not reuse its virtual environment from Windows Python.

### Usage

```bash
# Build and save the index.
uv run python -m src index --max_chunk_size=2000

# Search one question.
uv run python -m src search "How does the KV cache work?" --k=5

# Answer one question.
uv run python -m src answer "What is the KV cache?" --k=5

# Retrieve a dataset and save structured JSON.
uv run python -m src search_dataset \
    --dataset_path="datasets_public/public/UnansweredQuestions/dataset_docs_public.json" \
    --k=10 \
    --save_directory="data/output/search_results/public"

# Evaluate those retrieval results.
uv run python -m src evaluate \
    --student_search_results_path="data/output/search_results/public/dataset_docs_public.json" \
    --dataset_path="datasets_public/public/AnsweredQuestions/dataset_docs_public.json"

# Generate answers from the saved results, without searching again.
uv run python -m src answer_dataset \
    --student_search_results_path="data/output/search_results/public/dataset_docs_public.json" \
    --save_directory="data/output/search_results_and_answer/public"
```

Repeat dataset commands with `dataset_code_public.json` for code evaluation. Outputs retain the input dataset's filename. Keep the source files unchanged between indexing, retrieval, and answer generation.

The Makefile provides `install`, `run`, `debug`, `clean`, and `lint`. Use `make lint` for code checks. Installation should remain portable; a machine-specific cache directory is not required by `uv sync`.

## System architecture

| Component | Files and responsibility |
|---|---|
| Entry point and CLI | `__main__.py`, `cli.py`: expose commands and handle command errors. |
| Data structures | `models.py`: validate questions, sources, chunks, and outputs with Pydantic. |
| Ingestion | `read_files.py`: discover and read `.py`, `.md`, and `.txt` files. |
| Segmentation | `chunking.py`: split documents and attach Python definition names. |
| Indexing and retrieval | `indexer.py`, `retriever.py`: persist BM25 and rank chunks. |
| JSON input/output | `datasets.py`: load datasets and save results. |
| Generation | `llm.py`, `pipeline.py`: load Qwen, build prompts, and generate answers. |
| Evaluation | `evaluation.py`: compare source locations and calculate recall@k. |

Documents pass through ingestion, chunking, indexing, retrieval, and generation. Evaluation compares retrieved locations with references independently of generated text. Batch search and answer outputs use the subject's Pydantic models.

## Chunking strategy

Source ranges use **end-exclusive character positions**, preserving original line endings. Source excerpts never exceed the configured maximum of 2,000 characters.

- **Python:** AST top-level class, function, and async-function starts define sections. Oversized sections are split near newlines, with overlap. Invalid syntax falls back to size/newline splitting. Decorators do not receive special splitting treatment.
- **Markdown/text:** prefer paragraph boundaries, then line boundaries in the latter half of a window; otherwise cut at the size limit.
- **Overlap:** `min(200, max_chunk_size // 5)` characters, applied within Python sections or between text windows.
- **Merging:** the latest implementation combines adjacent, non-overlapping chunks from the same file whenever they fit.
- **Metadata:** `ast.walk()` finds class and function names, including methods, associated with each chunk. These names enrich indexing without changing original text or offsets.

## Retrieval method

Both queries and searchable chunks are lowercased and tokenized using `[a-z0-9]+`. Underscores and punctuation separate terms; CamelCase is not split.

BM25 uses `k1=1.5` and `b=0.75`. It combines term rarity (IDF), within-chunk term frequency with diminishing returns, and length normalization. Python metadata participates in those statistics. `get_scores()` calculates scores, and `heapq.nlargest()` selects the highest-ranked positions. An all-zero score array produces no results.

The index and ordered chunks are saved together in `data/processed/bm25.pkl`. Only trusted pickle files should be loaded. Rebuild after corpus or indexing changes.

## Generation and evaluation

Qwen receives numbered excerpts and instructions to answer from their evidence, preserve exact identifiers, and acknowledge missing information. Generation is greedy, thinking is disabled, and the default output limit is 200 new tokens. The pipeline reserves output space and removes lowest-ranked excerpts when the templated input exceeds its budget, capped at 4,096 input tokens.

A retrieved source matches a reference when paths are identical and character-range **IoU is at least 0.05**:

```text
IoU = intersection length / union length
Question recall@k = reference sources found in top k / reference sources
Dataset recall@k = average question recall@k
```

Missing predictions count as zero. Generated answer quality is assessed separately. The application does not call or import the official moulinette.

## Performance analysis

Recorded official retrieval exam, **24 November 2026**, at 2,000 characters:

| Check | Result | Requirement |
|---|---:|---:|
| Indexing | 38 s | ≤ 300 s |
| Retrieval, 200 questions | 25 s | ≤ 90 s |
| Private docs recall@5 | 80% | ≥ 80% |
| Private code recall@5 | 52% | ≥ 50% |

**Official result: PASS, 4/4.** Public recall@5 was 82% for docs and 56.6% for code in the corresponding configuration.

These measurements precede the latest change to merge every adjacent pair that fits; that change requires rebenchmarking. Earlier 1,000-character experiments reached 83% public docs and 61.6% public code recall@5, but other configuration differences prevent attributing the change solely to chunk size.

## Design decisions and challenges

- **BM25:** fast and effective for identifiers. TF-IDF was tested but reduced documentation recall to 68%, so BM25 was retained.
- **Structural context:** Python names improve matching for excerpts from inside large classes. Overlap preserves boundary context but increases repeated content.
- **Stable locations:** exact corpus paths and preserved line endings prevent source-matching errors.
- **Runtime:** stage timings identified slow WSL file reads. A compatible CUDA 12.6 PyTorch build resolved the development machine's GPU initialization failure.
- **Generation limits:** the small model can omit facts or misread tables. A two-stage evidence-extraction experiment added latency and sometimes discarded useful information, so direct generation was retained.
- **Errors:** unreadable files are reported and skipped; unrecoverable command errors produce a readable message and failure exit status.

## Verification

With the supplied evaluation assets installed:

```bash
bash exams/scripts/exam_retrieval.sh \
    --student-path . --moulinette-path ./moulinette/moulinette-ubuntu
bash exams/scripts/exam_answer.sh \
    --student-path . --moulinette-path ./moulinette/moulinette-ubuntu
bash exams/scripts/exam_edge_cases.sh --student-path .
make lint
```

The retrieval exam expects private datasets under `data/datasets/private/`. The answer exam requires manual judgment of at least two satisfactory answers out of three. Edge-case checks previously passed 4/4. Rerun checks after final edits; those prior results do not certify later changes.

Keep external corpora, datasets, weights, virtual environments, generated indexes, and outputs out of Git. Submit `src/`, dependency/configuration files, the Makefile, and this file as **`README.md`** at the root.

## Resources

- [Python AST](https://docs.python.org/3/library/ast.html) and [pickle](https://docs.python.org/3/library/pickle.html)
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) and [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) and [Transformers](https://huggingface.co/docs/transformers/)
- [Pydantic](https://docs.pydantic.dev/), [Python Fire](https://google.github.io/python-fire/), [tqdm](https://tqdm.github.io/), and [uv](https://docs.astral.sh/uv/)
- The supplied project subject and correction sheet.

**AI assistance:** ChatGPT helped explain concepts, draft and revise ingestion/chunking/indexing/retrieval/evaluation code, debug CUDA and typing errors, design tests, compare results, refine prompts, and write documentation. Suggestions were iterated on with local tests; the author remains responsible for reviewing, understanding, and validating the submitted work.
