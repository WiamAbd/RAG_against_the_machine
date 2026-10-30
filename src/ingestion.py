"""Discover and read repository files."""

from pathlib import Path


def discover_files(raw_directory: str) -> list[Path]:
    """Find Python, Markdown, and text files inside a directory."""
    root = Path(raw_directory)

    if not root.is_dir():
        raise ValueError(f"Directory does not exist: {raw_directory}")

    files: list[Path] = []

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".py", ".md", ".txt"}:
            files.append(path)

    return sorted(files)

def read_file(file_path: Path) -> str:
    """Read a UTF-8 file while preserving its original line endings."""
    with file_path.open("r", encoding="utf-8", newline="") as file:
        return file.read()

def load_documents(raw_directory: str) -> list[dict[str, str]]:
    """Load discovered files, reporting and skipping unreadable files."""
    files = discover_files(raw_directory)
    documents: list[dict[str, str]] = []

    for file_path in files:
        try:
            text = read_file(file_path)
        except (OSError, UnicodeError) as error:
            print(f"Skipping {file_path}: {error}")
            continue

        documents.append({
            "file_path": str(file_path),
            "text": text,
        })

    return documents
