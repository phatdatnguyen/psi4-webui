"""Validate names and keep user-selected files inside their working directory."""
import ntpath
from pathlib import Path


def validate_name(name: str, label: str = "File name") -> str:
    """Accept a single path component, on both Unix and Windows."""
    if (not isinstance(name, str) or not name.strip() or name in {".", ".."}
            or any(character in name for character in ("/", "\\", "\0"))
            or ntpath.splitdrive(name)[0]):
        raise ValueError(f"{label} must be a name without a directory path.")
    return name


def file_in_directory(directory: str, name: str) -> str:
    """Resolve a file path, rejecting traversal and links outside the directory."""
    if not directory:
        raise ValueError("Please open a working directory first.")
    validate_name(name)
    root = Path(directory).resolve()
    path = root / name
    if path.resolve().parent != root:
        raise ValueError("The file must be inside the working directory.")
    return str(path)
