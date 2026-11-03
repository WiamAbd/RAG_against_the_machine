"""Split repository documents into searchable chunks."""

from .models import Chunk
import ast

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
    """Split text into chunks, preferring paragraph and line boundaries."""
    if not 1 <= max_chunk_size <= 2000:
        raise ValueError("max_chunk_size must be between 1 and 2000.")

    chunks: list[Chunk] = []
    start = 0

    while start < len(text):
        end = min(start + max_chunk_size, len(text))

        if end < len(text):
            paragraph = text.rfind("\n\n", start, end)

            if paragraph > start:
                end = paragraph + 2
            else:
                line = text.rfind("\n", start, end)

                if line > start:
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