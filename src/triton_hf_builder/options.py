"""Validated generation inputs, independent of filesystems and inference engines."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class BuildOptions:
    name: str
    image: str
    device: str = "gpu"
    task: str | None = None
    tensor_parallel_size: int = 1
    max_new_tokens: int = 128
    max_model_len: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]*", self.name
        ):
            raise ValueError("Model name must contain only letters, digits, underscores or hyphens")
        if self.device not in ("cpu", "gpu"):
            raise ValueError("Device must be cpu or gpu")
        if (
            not isinstance(self.image, str)
            or not self.image
            or any(c.isspace() for c in self.image)
        ):
            raise ValueError("A target serving image is required")
        leaf = self.image.rsplit("/", 1)[-1]
        if "@" in leaf:
            valid = re.fullmatch(r"[^@]+@sha256:[0-9a-fA-F]{64}", leaf)
        else:
            repository, separator, tag = leaf.rpartition(":")
            valid = (
                repository
                and separator
                and tag != "latest"
                and re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", tag)
            )
        if not valid:
            raise ValueError("Use an explicit serving image tag or sha256 digest, not latest")
        for label, value in (
            ("tensor_parallel_size", self.tensor_parallel_size),
            ("max_new_tokens", self.max_new_tokens),
            ("max_model_len", self.max_model_len),
        ):
            if value is None and label == "max_model_len":
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f"{label} must be a positive integer")
