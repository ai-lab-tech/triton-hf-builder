"""Resolve serving requirement declarations without loading or installing packages."""

from collections.abc import Sequence
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from .dependency_profiles import image_corrections
from .runtimes import RuntimeAdapter


def image_requirements(runtime: str, image: str) -> tuple[str, ...]:
    """Return reviewed corrections for an exact image, also used by serving tests."""
    return image_corrections(runtime, image)


def _parse_requirement(value: str) -> Requirement:
    if not isinstance(value, str) or any(c in value for c in ("\n", "\r")):
        raise ValueError("Each requirement must be one package specification")
    try:
        parsed = Requirement(value)
    except InvalidRequirement as error:
        raise ValueError(f"Invalid requirement: {value}") from error
    if parsed.url:
        raise ValueError("Use package version specifications, not URL requirements")
    return parsed


def resolve_requirements(
    adapter: RuntimeAdapter,
    source: Path,
    extras: Sequence[str],
    *,
    image: str = "",
    profile: dict | None = None,
    transformers_version: str | None = None,
) -> list[str]:
    """Apply defaults, profile, model Transformers version and explicit overrides."""
    if isinstance(extras, (str, bytes)) or not isinstance(extras, Sequence):
        raise ValueError("Requirements must be a sequence of package specifications")
    defaults = list(adapter.default_requirements)
    if any((source / name).is_file() for name in ("tokenizer.model", "spiece.model")):
        defaults.append("sentencepiece")
    # Bundled profiles already include their image corrections. Apply those
    # separately only when no profile covers this image/device combination.
    defaults.extend(profile["requirements"] if profile else image_requirements(adapter.name, image))
    parsed_extras = [_parse_requirement(value) for value in extras]
    unconditional: dict[str, str] = {}
    conditional: dict[str, str] = {}
    for parsed in [*(_parse_requirement(value) for value in defaults), *parsed_extras]:
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
    explicit_transformers = any(
        canonicalize_name(parsed.name) == "transformers" and parsed.marker is None
        for parsed in parsed_extras
    )
    if (
        adapter.name == "transformers"
        and transformers_version is not None
        and not explicit_transformers
    ):
        try:
            if not isinstance(transformers_version, str):
                raise InvalidVersion(str(transformers_version))
            version = Version(transformers_version)
        except InvalidVersion as error:
            raise ValueError(
                "Model transformers_version must be a valid package version; "
                "pass an explicit transformers requirement to override it"
            ) from error
        unconditional["transformers"] = f"transformers=={version}"
    return [*unconditional.values(), *conditional.values()]


TORCH_PACKAGES = frozenset({"torch", "torchvision", "torchaudio"})


def serving_torch_requirements(requirements: list[str], device: str) -> tuple[list[str], list[str]]:
    """Select CPU wheels from exact pins; preserve explicitly selected GPU builds.

    Generation never inspects the builder host's installed packages. Unknown CPU
    targets need an explicit version instead of guessing a compatible release.
    """
    resolved = []
    sources = []
    for value in requirements:
        requirement = Requirement(value)
        name = canonicalize_name(requirement.name)
        if name not in TORCH_PACKAGES:
            resolved.append(value)
            continue
        pins = [s for s in requirement.specifier if s.operator == "==" and "*" not in s.version]
        if device == "cpu" and len(pins) != 1:
            raise ValueError(
                f"CPU serving requires an exact {name} version; "
                f"select a supported image profile or pass {name}==<version>"
            )
        if len(pins) == 1:
            pin = pins[0]
            version = Version(pin.version)
            local = version.local
            if device == "gpu" and local == "cpu":
                raise ValueError(f"GPU serving cannot use {name}=={version}; select a GPU build")
            specifications = [str(s) for s in requirement.specifier if s != pin]
            if device == "cpu" and local is None:
                local = "cpu"
                specifications.append(f"=={version.public}+cpu")
            else:
                specifications.append(str(pin))
                if device == "gpu" and local is None:
                    specifications.append(f"!={version.public}+cpu")
            requirement.specifier = type(requirement.specifier)(",".join(specifications))
            if local == "cpu" or (local and local.startswith("cu") and local[2:].isdigit()):
                source = f"https://download.pytorch.org/whl/{local}/{name}/"
                if source not in sources:
                    sources.append(source)
        resolved.append(str(requirement))
    return resolved, sources
