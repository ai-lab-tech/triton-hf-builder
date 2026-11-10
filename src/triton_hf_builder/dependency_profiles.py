"""Select bundled dependency profiles without network or engine imports."""

import json
from importlib.resources import files


def _catalog() -> dict:
    return json.loads(
        files("triton_hf_builder").joinpath("dependency_profiles.json").read_text(encoding="utf-8")
    )


def dependency_profile(runtime: str, image: str, device: str) -> dict | None:
    # A digest selects the image independently of an accompanying tag. Only an
    # explicitly cataloged reference may select a profile; never discard it.
    for profile in _catalog()["profiles"]:
        if (profile["runtime"], profile["image"], profile["device"]) == (runtime, image, device):
            return profile
    return None


def image_corrections(runtime: str, image: str) -> tuple[str, ...]:
    for correction in _catalog()["image_corrections"]:
        if (correction["runtime"], correction["image"]) == (runtime, image):
            return tuple(correction["requirements"])
    return ()


def architecture_profile(runtime: str, image: str) -> dict | None:
    for profile in _catalog().get("architecture_profiles", []):
        if (profile["runtime"], profile["image"]) == (runtime, image):
            return profile
    return None
