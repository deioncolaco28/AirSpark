"""
Path management utilities for AirSpark.
Ensures cross-platform (Windows / macOS / Linux) safe path operations.
"""
from pathlib import Path
from typing import Union

# Base project directory is the parent of the `app` package directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return PROJECT_ROOT


def resolve_path(relative_or_absolute_path: Union[str, Path]) -> Path:
    """
    Resolve a path relative to the project root if it is not already absolute.
    """
    p = Path(relative_or_absolute_path)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """Ensure directory exists and return resolved Path object."""
    p = resolve_path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    return p
