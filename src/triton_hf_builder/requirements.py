"""Resolve serving requirement declarations without loading or installing packages."""

from collections.abc import Sequence
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from .runtimes import RuntimeAdapter


def resolve_requirements(adapter: RuntimeAdapter, source: Path, extras: Sequence[str]) -> list[str]:
    if isinstance(extras, (str, bytes)) or not isinstance(extras, Sequence):
        raise ValueError("Requirements must be a sequence of package specifications")
    defaults = list(adapter.default_requirements)
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
