"""Compare checkpoint architectures with versioned, bundled engine documentation."""

from .dependency_profiles import architecture_profile


def check_architectures(runtime: str, image: str, architectures: tuple[str, ...]) -> dict:
    """Return documentation evidence, never a claim of successful serving."""
    result = {
        "status": "unknown",
        "runtime": runtime,
        "image": image,
        "architectures": [],
        "warnings": [],
    }
    if runtime == "transformers":
        result["status"] = "not_checked"
        return result
    profile = architecture_profile(runtime, image)
    if profile is None:
        result["warnings"].append(
            f"Architecture support is unknown: no bundled documentation for {runtime} image {image}"
        )
        return result
    for field in ("engine_version", "model_source", "version_match"):
        result[field] = profile[field]
    if profile["architecture_status"] != "upstream_documented":
        result["warnings"].append(
            f"Architecture support is unknown: versioned {runtime} model documentation is unavailable"
        )
        return result
    if not architectures:
        result["warnings"].append(
            "Architecture support is unknown: model config.json has no architectures"
        )
        return result
    rows = {row["architecture"]: row for row in profile["architectures"]}
    for name in architectures:
        row = rows.get(name)
        if row is None:
            entry = {"architecture": name, "status": "not_documented"}
            result["warnings"].append(
                f"Architecture {name} is not listed in the bundled {runtime} documentation "
                f"for {image}; support is unverified, not necessarily unavailable"
            )
        else:
            entry = {
                **row,
                "status": (
                    "upstream_documented"
                    if row["builder_scope"] == "text_generation_candidate"
                    else "outside_builder_scope"
                ),
            }
            if entry["status"] == "outside_builder_scope":
                result["warnings"].append(
                    f"Architecture {name} is documented for {runtime}, but is outside "
                    "the catalog's builder text-generation scope"
                )
            if row["modeling_backend"] == "transformers":
                result["warnings"].append(
                    f"Architecture {name} is documented through {runtime}'s Transformers "
                    "modeling backend; verify that backend in the destination image"
                )
        result["architectures"].append(entry)
    statuses = {entry["status"] for entry in result["architectures"]}
    result["status"] = next(
        status
        for status in ("upstream_documented", "outside_builder_scope", "not_documented")
        if status in statuses
    )
    if profile["version_match"] == "upstream_base_only":
        result["warnings"].append(
            "Architecture documentation matches the upstream base engine version only, "
            "not the exact NVIDIA engine build"
        )
    return result
