"""Generate serving artifacts without importing ML frameworks or loading weights."""

from ._version import __version__
from .builder import BuildResult, generate_repository
from .inspector import ModelInfo, inspect_model

__all__ = ["BuildResult", "ModelInfo", "generate_repository", "inspect_model", "__version__"]
