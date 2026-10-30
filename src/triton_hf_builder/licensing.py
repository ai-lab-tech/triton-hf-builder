"""Preserve checkpoint attribution and license generated support code."""

from importlib.metadata import distribution
from pathlib import Path

_ATTRIBUTION_NAMES = (
    "license",
    "licence",
    "notice",
    "copying",
    "copyright",
    "authors",
    "readme",
    "model_card",
    "model-card",
)
_DOCUMENT_SUFFIXES = {"", ".txt", ".md", ".markdown", ".rst"}


def is_attribution_file(name: str) -> bool:
    """Recognize root attribution documents without accepting executable files."""
    path = Path(name)
    if path.suffix.casefold() not in _DOCUMENT_SUFFIXES:
        return False
    stem = path.stem.casefold()
    return any(
        stem == base or stem.startswith((base + "-", base + "_", base + "."))
        for base in _ATTRIBUTION_NAMES
    )


def generator_license() -> str:
    """Read the license installed by the package's license-files metadata."""
    content = distribution("triton-hf-builder").read_text("licenses/LICENSE")
    if not content:
        raise RuntimeError("The installed triton-hf-builder distribution is missing its LICENSE")
    return content
