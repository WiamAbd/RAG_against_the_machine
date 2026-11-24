"""Split repository documents into searchable chunks."""

from .models import Chunk
import ast
from pathlib import Path
from tqdm import tqdm


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
    """Split text into overlapping chunks with original character offsets."""

    # Default: 20% overlap, capped at 200 characters.
    overlap_size = min(200, max_chunk_size // 5)

    chunks: list[Chunk] = []
    start = 0

    while start < len(text):
        end = min(start + max_chunk_size, len(text))

        if end < len(text):
            # Avoid short chunks and guarantee progress after overlap.
            minimum_end = start + max(1, max_chunk_size // 2)

            paragraph = text.rfind("\n\n", start, end)

            if paragraph >= minimum_end:
                end = paragraph + 2
            else:
                line = text.rfind("\n", start, end)

                if line >= minimum_end:
                    end = line + 1

        chunks.append(create_chunk(file_path, text, start, end))

        if end == len(text):
            break

        start = end - overlap_size

    return chunks


def chunk_python(
    file_path: str,
    text: str,
    max_chunk_size: int = 2000,
) -> list[Chunk]:
    """Split Python, overlapping oversized sections"""

    overlap_size = min(200, max_chunk_size // 5)
    # find end of line character
    line_offsets = [0]
    position = 0

    for line in text.splitlines(keepends=True):
        position += len(line)
        line_offsets.append(position)

    boundaries = {0, len(text)}

    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = None

    if tree is not None:
        for node in tree.body:
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                first_line = node.lineno

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

                if line_break > start:
                    end = line_break + 1

            chunks.append(create_chunk(file_path, text, start, end))

            if end == section_end:
                break

            # Overlap within this AST section only.
            # Ensure that the next chunk always advances.
            start = max(start + 1, end - overlap_size)

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


def add_python_context(
    text: str,
    chunks: list[Chunk],
) -> list[Chunk]:
    """Attach names of Python definitions overlapping each chunk."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # Keep the chunks searchable even if Python parsing fails.
        return chunks

    line_offsets = [0]

    for line in text.splitlines(keepends=True):
        line_offsets.append(line_offsets[-1] + len(line))

    definitions: list[tuple[int, int, str, str]] = []

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue

        if node.end_lineno is None:
            continue

        first_line = node.lineno

        if node.decorator_list:
            first_line = min(
                first_line,
                node.decorator_list[0].lineno,
            )

        start = line_offsets[first_line - 1]
        end = line_offsets[node.end_lineno]

        kind = "class" if isinstance(node, ast.ClassDef) else "function"

        definitions.append((start, end, kind, node.name))

    for chunk in chunks:
        class_names: set[str] = set()
        function_names: set[str] = set()

        for start, end, kind, name in definitions:
            overlaps = (
                start < chunk.last_character_index
                and end > chunk.first_character_index
            )

            if not overlaps:
                continue

            if kind == "class":
                class_names.add(name)
            else:
                function_names.add(name)

        chunk.class_names = sorted(class_names)
        chunk.function_names = sorted(function_names)

    return chunks


def chunk_documents(
    documents: list[dict[str, str]],
    max_chunk_size: int = 2000,
    min_chunk_size: int = 200,
) -> list[Chunk]:
    """Chunk Python and Markdown documents and merge short neighbors."""
    if type(min_chunk_size) is not int:
        raise ValueError("Chunk sizes must be integers.")

    if not 1 <= min_chunk_size <= 2000:
        raise ValueError(
            "Sizes must satisfy: 1 <= min_chunk_size <= 2000"
        )

    all_chunks: list[Chunk] = []

    for document in tqdm(
        documents,
        desc="Chunking documents",
        unit="file",
    ):
        file_path = document["file_path"]
        text = document["text"]
        extension = Path(file_path).suffix.lower()

        if extension == ".py":
            chunks = chunk_python(file_path, text, max_chunk_size)
        elif extension in (".md", ".txt"):
            chunks = chunk_text(file_path, text, max_chunk_size)
        else:
            continue

        chunks = merge_small_chunks(
            chunks,
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
        )
        if extension == ".py":
            chunks = add_python_context(text, chunks)
        all_chunks.extend(chunks)

    return all_chunks
