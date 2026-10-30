"""Build a sourced catalog and a compact README view of manufacturer claims."""

import re

from .parsing import image_rows, model_rows
from .sources import MissingSource

BEGIN = "<!-- compatibility:begin -->"
END = "<!-- compatibility:end -->"


def build_catalog(sources, policy):
    text, matrix_source = sources.document(
        "triton-inference-server/server", "main", "docs/introduction/compatibility.md"
    )
    rows = image_rows(text)
    selected = []
    count = policy["recent_per_runtime"]
    if type(count) is not int or not 1 <= count <= 3:
        raise ValueError("Track between one and three recent images per runtime")
    for runtime in ("vllm", "tensorrt-llm"):
        candidates = sorted(
            (row for row in rows if row["runtime"] == runtime),
            key=lambda row: row["release"],
            reverse=True,
        )
        releases = set(policy["retain_releases"]) | {row["release"] for row in candidates[:count]}
        if not set(policy["retain_releases"]).issubset({r["release"] for r in candidates}):
            raise ValueError(f"Retained release missing from {runtime} image matrix")
        selected.extend(row for row in candidates if row["release"] in releases)
    for row in selected:
        row.update(
            architectures=[],
            architecture_status="unknown",
            model_source=None,
            builder_test_status="not_tested",
            quantization_status="not_evaluated",
            gpu_support_status="not_evaluated",
        )
        # NVIDIA suffixes are retained above; matching a base version does not
        # prove that an NVIDIA-patched build is identical to upstream.
        version = row["engine_version"]
        match = re.fullmatch(r"(\d+\.\d+\.\d+(?:\.post\d+)?)(\+[^\s]+)?", version)
        if not match:
            row["reason"] = "No unambiguous stable upstream release mapping"
            continue
        base, suffix = match.groups()
        row["upstream_version"] = base
        row["version_match"] = "upstream_base_only" if suffix else "release_version"
        runtime = row["runtime"]
        repo, path = (
            ("vllm-project/vllm", "docs/models/supported_models.md")
            if runtime == "vllm"
            else ("NVIDIA/TensorRT-LLM", "docs/source/models/supported-models.md")
        )
        try:
            text, source = sources.document(repo, f"v{base}", path)
        except MissingSource:
            row["reason"] = "Exact upstream release document unavailable; no latest-doc fallback"
            continue
        row["architectures"] = model_rows(text, runtime)
        row["model_source"] = source
        row["architecture_status"] = "upstream_documented"
    return {
        "schema_version": 1,
        "policy": policy,
        "image_matrix_source": matrix_source,
        "scope": "Text generation candidates only; consult versioned sources for all conditions. "
        "Architecture listing does not verify quantization, GPU, custom code or adapter compatibility.",
        "images": selected,
    }


def render_table(catalog):
    lines = [
        BEGIN,
        "",
        "Architecture support reported by the backend projects; "
        "individual models may require additional configuration.",
        "",
        "| Triton image | Backend | Supported architectures |",
        "| --- | --- | --- |",
    ]
    for row in catalog["images"]:
        tag = row["image"].split(":", 1)[1]
        source = row["model_source"]
        count = sum(a["builder_scope"] == "text_generation_candidate" for a in row["architectures"])
        reference = f"[{count} candidates]({source['url']})" if source else "Unknown"
        if row.get("version_match") == "upstream_base_only":
            reference += " †"
        backend = "vLLM" if row["runtime"] == "vllm" else "TensorRT-LLM"
        lines.append(f"| `{tag}` | {backend} | {reference} |")
    lines += [
        "",
        f"Source: [NVIDIA image matrix]({catalog['image_matrix_source']['url']}). "
        "† Upstream base version; NVIDIA build differs.",
        "",
        "[Full catalog](compatibility.json): architecture names, exact engine versions, "
        "CUDA and reported driver versions. See the linked backend documentation "
        "for quantization, GPU and feature requirements.",
        "",
    ]
    return "\n".join([*lines, END])


def update_readme(readme, catalog):
    if readme.count(BEGIN) != 1 or readme.count(END) != 1:
        raise ValueError("README must contain exactly one compatibility marker pair")
    start, end = readme.index(BEGIN), readme.index(END)
    if end < start:
        raise ValueError("Invalid compatibility marker order")
    return readme[:start] + render_table(catalog) + readme[end + len(END) :]


def change_summary(previous, current):
    old = {row["image"]: row for row in previous.get("images", [])}
    new = {row["image"]: row for row in current["images"]}
    lines = [
        "Refresh manufacturer-documented compatibility; no serving tests were run.",
        "",
        "| Image | Change | Architecture entries added / removed |",
        "| --- | --- | --- |",
    ]
    for image in sorted(old.keys() | new.keys()):
        before, after = old.get(image), new.get(image)
        if before == after:
            continue
        a = {r["architecture"] for r in before["architectures"]} if before else set()
        b = {r["architecture"] for r in after["architectures"]} if after else set()
        if before is None:
            change = f"Added: {after['engine_version']}"
        elif after is None:
            change = "Removed by tracking policy"
        elif before["engine_version"] != after["engine_version"]:
            change = f"Engine {before['engine_version']} → {after['engine_version']}"
        else:
            change = "Source, metadata or architecture update"
        lines.append(f"| `{image}` | {change} | +{len(b - a)} / -{len(a - b)} |")
    if old == new:
        lines.append("| — | No image-entry changes | — |")
    lines += [
        "",
        "Review `compatibility.json` and the README diff for versioned source links, "
        "NVIDIA build differences and architecture scope. Quantization/GPU support is not "
        "inferred. New entries are not automatically recommended or marked tested.",
        "",
        "The workflow validates offline consistency, unit tests, lint and formatting before "
        "opening this PR. No automatic merge. PRs made with GITHUB_TOKEN may not trigger ordinary PR CI.",
        "",
    ]
    return "\n".join(lines)
