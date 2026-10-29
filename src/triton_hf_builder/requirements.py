"""Resolve serving requirement declarations without loading or installing packages."""

from collections.abc import Sequence
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from .runtimes import RuntimeAdapter


def image_requirements(runtime: str, image: str) -> tuple[str, ...]:
    """Narrow fixes for identified serving images, shared with derived-image tests.

    NVIDIA documents the PartReasoningText import failure and an SDK upgrade:
    https://github.com/triton-inference-server/server/blob/main/docs/getting_started/llm.md
    The required type exists in the verified OpenAI SDK v2.8.1 release. This is
    our reproducible pin, not an NVIDIA-mandated version or a full serving test.
    """
    if runtime == "tensorrt-llm" and image.split("@", 1)[0] == (
        "nvcr.io/nvidia/tritonserver:26.06-trtllm-python-py3"
    ):
        return ("openai==2.8.1",)
    return ()


def resolve_requirements(
    adapter: RuntimeAdapter, source: Path, extras: Sequence[str], *, image: str = ""
) -> list[str]:
    if isinstance(extras, (str, bytes)) or not isinstance(extras, Sequence):
        raise ValueError("Requirements must be a sequence of package specifications")
    defaults = [*adapter.default_requirements, *image_requirements(adapter.name, image)]
    if any((source / name).is_file() for name in ("tokenizer.model", "spiece.model")):
        defaults.append("sentencepiece")
    unconditional: dict[str, str] = {}
    conditional: dict[str, str] = {}
    for value in [*defaults, *extras]:
        if not isinstance(value, str) or any(c in value for c in ("\n", "\r")):
            raise ValueError("Each requirement must be one package specification")
        try:
            parsed = Requirement(value)
        except InvalidRequirement as error:
            raise ValueError(f"Invalid requirement: {value}") from error
        if parsed.url:
            raise ValueError("Use package version specifications, not URL requirements")
        key = canonicalize_name(parsed.name)
        if key in adapter.managed_packages:
            raise ValueError(f"{key} must come from the selected {adapter.name} image")
        normalized = str(parsed)
        if parsed.marker:
            # Keep platform alternatives and the mandatory default. An inactive
            # marker must never remove an essential runtime dependency.
            conditional[normalized] = normalized
        else:
            unconditional[key] = normalized
    return [*unconditional.values(), *conditional.values()]
