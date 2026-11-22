"""Split repository documents into searchable chunks."""

from .models import Chunk
import ast
from pathlib import Path
from time import perf_counter
def create_chunk(
    file_path: str,
    text: str,
    start: int,
    end: int,
) -> Chunk:
    """Create a chunk using an end-exclusive character range."""
    return Chunk(
        file_path=file_path,
        first_character_index=start,
        last_character_index=end,
        text=text[start:end],
    )

def chunk_text(
    file_path: str,
    text: str,
    max_chunk_size: int = 2000,
) -> list[Chunk]:
    """Split text while avoiding unnecessarily short intermediate chunks."""
    if type(max_chunk_size) is not int:
        raise ValueError("max_chunk_size must be an integer.")

    if not 1 <= max_chunk_size <= 2000:
        raise ValueError("max_chunk_size must be between 1 and 2000.")

    chunks: list[Chunk] = []
    start = 0

    while start < len(text):
        end = min(start + max_chunk_size, len(text))

        if end < len(text):
            minimum_end = start + max(1, max_chunk_size // 2)

            paragraph = text.rfind("\n\n", start, end)

            if paragraph >= minimum_end:
                end = paragraph + 2
            else:
                line = text.rfind("\n", start, end)

                if line >= minimum_end:
                    end = line + 1

        chunks.append(create_chunk(file_path, text, start, end))
        start = end

    return chunks

def chunk_python(
    file_path: str,
    text: str,
    max_chunk_size: int = 2000,
) -> list[Chunk]:
    """Split Python at top-level definitions and respect the size limit."""
    if type(max_chunk_size) is not int:
        raise ValueError("max_chunk_size must be an integer.")

    if not 1 <= max_chunk_size <= 2000:
        raise ValueError("max_chunk_size must be between 1 and 2000.")

    # Record the character position where each line begins.
    line_offsets = [0]
    position = 0

    for line in text.splitlines(keepends=True):
        position += len(line)
        line_offsets.append(position)

    boundaries = {0, len(text)}

    try:
        tree = ast.parse(text)
    except SyntaxError:
        # If parsing fails, fall back to splitting at line boundaries.
        tree = None

    if tree is not None:
        for node in tree.body:
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                first_line = node.lineno

                # Include decorators with their function or class.
                if node.decorator_list:
                    first_line = min(
                        first_line,
                        node.decorator_list[0].lineno,
                    )

                boundaries.add(line_offsets[first_line - 1])

    ordered_boundaries = sorted(boundaries)
    chunks: list[Chunk] = []

    for index in range(len(ordered_boundaries) - 1):
        start = ordered_boundaries[index]
        section_end = ordered_boundaries[index + 1]

        while start < section_end:
            end = min(start + max_chunk_size, section_end)

            if end < section_end:
                line_break = text.rfind("\n", start, end)

                if line_break >= start:
                    end = line_break + 1

            chunks.append(create_chunk(file_path, text, start, end))
            start = end

    return chunks

def merge_small_chunks(
    chunks: list[Chunk],
    max_chunk_size: int = 2000,
    min_chunk_size: int = 200,
) -> list[Chunk]:
    """Merge adjacent chunks when either is short and their total fits."""
    if type(max_chunk_size) is not int or type(min_chunk_size) is not int:
        raise ValueError("Chunk sizes must be integers.")

    if not 1 <= min_chunk_size <= max_chunk_size <= 2000:
        raise ValueError(
            "Sizes must satisfy: 1 <= min_chunk_size "
            "<= max_chunk_size <= 2000."
        )

    merged: list[Chunk] = []

    for chunk in chunks:
        if merged:
            previous = merged[-1]

            same_file = previous.file_path == chunk.file_path
            adjacent = (
                previous.last_character_index
                == chunk.first_character_index
            )
            either_short = (
                len(previous.text) < min_chunk_size
                or len(chunk.text) < min_chunk_size
            )
            fits = (
                len(previous.text) + len(chunk.text)
                <= max_chunk_size
            )

            if same_file and adjacent and either_short and fits:
                merged[-1] = Chunk(
                    file_path=previous.file_path,
                    first_character_index=previous.first_character_index,
                    last_character_index=chunk.last_character_index,
                    text=previous.text + chunk.text,
                )
                continue

        merged.append(chunk)

    return merged

def chunk_documents(
    documents: list[dict[str, str]],
    max_chunk_size: int = 2000,
    min_chunk_size: int = 200,
) -> list[Chunk]:
    """Chunk Python and Markdown documents and merge short neighbors."""
    if type(max_chunk_size) is not int or type(min_chunk_size) is not int:
        raise ValueError("Chunk sizes must be integers.")

    if not 1 <= min_chunk_size <= max_chunk_size <= 2000:
        raise ValueError(
            "Sizes must satisfy: 1 <= min_chunk_size "
            "<= max_chunk_size <= 2000."
        )

    all_chunks: list[Chunk] = []

    for document in documents:
        file_path = document["file_path"]
        text = document["text"]
        extension = Path(file_path).suffix.lower()

        if extension == ".py":
            chunks = chunk_python(file_path, text, max_chunk_size)
        elif extension == ".md":
            chunks = chunk_text(file_path, text, max_chunk_size)
        else:
            continue

        chunks = merge_small_chunks(
            chunks,
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
        )

        all_chunks.extend(chunks)

    return all_chunks

if __name__ == "__main__":
    from .read_files import load_documents

    try:
        start = perf_counter()

        documents = load_documents("data/raw")
        reading_end = perf_counter()

        chunks = chunk_documents(documents)
        chunking_end = perf_counter()

        print(f"Loaded {len(documents)} documents.")
        print(f"Created {len(chunks)} chunks.")
        print(f"Reading: {reading_end - start:.2f} seconds")
        print(f"Chunking: {chunking_end - reading_end:.2f} seconds")
        print(f"Total: {(chunking_end - start)/60:.2f} minutes")

        for chunk in chunks[:5]:
            print(
                chunk.file_path,
                chunk.first_character_index,
                chunk.last_character_index,
                f"{len(chunk.text)} characters",
            )

        assert all(0 < len(chunk.text) <= 2000 for chunk in chunks)

        print("All chunk sizes are valid.")

    except Exception as error:
        print(f"Error: {error}")