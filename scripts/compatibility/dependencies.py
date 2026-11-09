"""Combine published image facts and reviewed dependency rules for offline use."""

import copy
from urllib.parse import urlsplit

from packaging.requirements import Requirement


def _validate_entry(entry):
    if entry.get("runtime") not in {"transformers", "vllm", "tensorrt-llm"}:
        raise ValueError("Unknown dependency runtime")
    image = entry.get("image")
    if not isinstance(image, str) or not image or any(c.isspace() for c in image):
        raise ValueError("Dependency entries need an explicit image reference")
    for field in ("requirements", "sources", "notes"):
        if not isinstance(entry.get(field), list):
            raise ValueError(f"Dependency {field} must be a list")
    for value in entry["requirements"]:
        if not isinstance(value, str) or "\n" in value or "\r" in value:
            raise ValueError("Dependency requirements must be single-line specifications")
        if Requirement(value).url:
            raise ValueError("Dependency requirements must not contain URLs")
    if not entry["sources"] or any(
        not isinstance(source, dict) or not source.get("url") for source in entry["sources"]
    ):
        raise ValueError("Dependency entries must include source URLs")
    if not all(isinstance(note, str) and note for note in entry["notes"]):
        raise ValueError("Dependency notes must be nonempty strings")


def _validate_profile(profile):
    _validate_entry(profile)
    if profile.get("device") not in {"cpu", "gpu"}:
        raise ValueError("Dependency profiles need a CPU or GPU device")
    if profile.get("status") not in {"manufacturer_documented", "example_tested"}:
        raise ValueError("Unknown dependency profile status")
    for field in ("package_indexes", "package_find_links"):
        urls = profile.get(field, [] if field == "package_find_links" else None)
        if not isinstance(urls, list):
            raise ValueError(f"Dependency {field} must be a list")
        for url in urls:
            if not isinstance(url, str) or any(c.isspace() for c in url):
                raise ValueError("Package indexes and wheel sources must be single HTTPS URLs")
            parsed = urlsplit(url)
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username
                or parsed.password
            ):
                raise ValueError(
                    "Package indexes and wheel sources must be HTTPS URLs without credentials"
                )


def _profile_key(profile):
    return profile["runtime"], profile["image"], profile["device"]


def _manufacturer_profile(row, source):
    return {
        "runtime": row["runtime"],
        "image": row["image"],
        "release": row["release"],
        "device": "gpu",
        "status": "manufacturer_documented",
        "requirements": [],
        "package_indexes": [],
        "environment": {
            "engine_version": row["engine_version"],
            "python_version": row["python_version"],
            "cuda_version": row["cuda_version"],
            "driver_version_reported": row["driver_version_reported"],
        },
        "sources": [copy.deepcopy(source)],
        "notes": [
            "Retain the engine and framework stack supplied by this image. "
            "Manufacturer documentation does not establish builder serving validation."
        ],
    }


def _merge_reviewed_profile(baseline, reviewed):
    for field in ("release", "environment"):
        if field in baseline and field in reviewed and reviewed[field] != baseline[field]:
            raise ValueError(f"Curated {field} disagrees with manufacturer image facts")
    profile = {**baseline, **copy.deepcopy(reviewed)}
    for field in ("sources", "notes"):
        profile[field] = [*baseline.get(field, []), *profile[field]]
    return profile


def _index_corrections(corrections):
    indexed = {}
    for correction in corrections:
        _validate_entry(correction)
        key = correction["runtime"], correction["image"]
        if key in indexed:
            raise ValueError("Duplicate image correction")
        indexed[key] = correction
    return indexed


def build_dependency_catalog(catalog, curated):
    if catalog.get("schema_version") != 1 or curated.get("schema_version") != 1:
        raise ValueError("Unsupported dependency catalog schema")
    profiles = {}
    for row in catalog["images"]:
        profile = _manufacturer_profile(row, catalog["image_matrix_source"])
        key = _profile_key(profile)
        if key in profiles:
            raise ValueError("Duplicate manufacturer dependency profile")
        profiles[key] = profile
    reviewed_keys = set()
    for reviewed in curated["profiles"]:
        _validate_profile(reviewed)
        key = _profile_key(reviewed)
        if key in reviewed_keys:
            raise ValueError("Duplicate curated dependency profile")
        reviewed_keys.add(key)
        profiles[key] = _merge_reviewed_profile(profiles.get(key, {}), reviewed)
    corrections = _index_corrections(curated["image_corrections"])
    for profile in profiles.values():
        correction = corrections.get((profile["runtime"], profile["image"]))
        if correction:
            for field in ("requirements", "sources", "notes"):
                profile[field].extend(copy.deepcopy(correction[field]))
        _validate_profile(profile)
    return {
        "schema_version": 1,
        "profiles": sorted(
            profiles.values(), key=lambda p: (p["image"], p["runtime"], p["device"])
        ),
        "image_corrections": copy.deepcopy(curated["image_corrections"]),
    }
