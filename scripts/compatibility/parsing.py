"""Strict parsers for supported, versioned upstream Markdown tables."""

import re


def tables(text):
    """Yield heading context, headers and rows; reject malformed table rows."""
    headings = []
    lines = text.splitlines()
    index = 0
    fence = None
    while index < len(lines):
        line = lines[index].strip()
        if line.startswith(("```", "~~~")):
            marker = line[:3]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            index += 1
            continue
        if fence:
            index += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            level = len(heading[1])
            headings = [(depth, title) for depth, title in headings if depth < level]
            headings.append((level, heading[2]))
        if line.startswith("|") and index + 1 < len(lines):
            separator = lines[index + 1].strip()
            if re.fullmatch(r"\|[\s:|\-]+\|", separator):
                headers = [cell.strip() for cell in line.strip("|").split("|")]
                rows = []
                index += 2
                while index < len(lines) and lines[index].strip().startswith("|"):
                    cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                    if len(cells) != len(headers):
                        raise ValueError("Upstream Markdown table shape changed")
                    rows.append(dict(zip(headers, cells)))
                    index += 1
                yield [title for _, title in headings], headers, rows
                continue
        index += 1


def image_rows(text):
    result = []
    engines = {
        "vllm": ("vllm-python-py3", "vLLM version"),
        "tensorrt-llm": ("trtllm-python-py3", "TensorRT-LLM version"),
    }
    for runtime, (suffix, version_column) in engines.items():
        matches = [
            rows
            for headings, headers, rows in tables(text)
            if f"Container Name: {suffix}" in headings and version_column in headers
        ]
        if len(matches) != 1 or not matches[0]:
            raise ValueError(f"Missing or ambiguous NVIDIA image matrix for {runtime}")
        seen = set()
        for row in matches[0]:
            release = row["Triton release version"]
            image = row["NGC Tag"]
            if not re.fullmatch(r"\d{2}\.\d{2}", release):
                raise ValueError(f"Unexpected Triton release: {release}")
            if image != f"nvcr.io/nvidia/tritonserver:{release}-{suffix}" or image in seen:
                raise ValueError(f"Unexpected or duplicate image: {image}")
            seen.add(image)
            result.append(
                {
                    "runtime": runtime,
                    "release": release,
                    "image": image,
                    "engine_version": row[version_column],
                    "cuda_version": row["CUDA version"],
                    "driver_version_reported": row["CUDA Driver version"],
                    "python_version": row["Python version"],
                }
            )
    return result


def model_rows(text, runtime):
    candidates = []
    for headings, headers, rows in tables(text):
        if not headers or headers[0] != "Architecture":
            continue
        if runtime == "vllm":
            eligible = (
                "List of Text-only Language Models" in headings and "Text Generation" in headings
            )
        else:
            eligible = headings == ["Supported Models"]
        if eligible:
            candidates.append(rows)
    if runtime == "tensorrt-llm" and "supported models for the PyTorch backend" not in text:
        raise ValueError("TensorRT-LLM document does not identify the PyTorch backend")
    valid_count = len(candidates) == 1 or (
        runtime == "vllm"
        and len(candidates) == 2
        and "Some models are supported only via the [Transformers modeling backend]" in text
    )
    if not valid_count or not all(candidates):
        raise ValueError(f"Cannot identify {runtime} text architecture table")
    result = []
    for table_index, row in ((i, row) for i, rows in enumerate(candidates) for row in rows):
        names = re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", row["Architecture"])
        if not names:
            raise ValueError("Unrecognized architecture name in upstream table")
        for name in names:
            # Conservative scope: the builder's engine adapters only generate text.
            # Conditional generation, rewards and classifiers remain out of scope.
            in_scope = name.endswith("ForCausalLM") or name == "GPT2LMHeadModel"
            result.append(
                {
                    "architecture": name,
                    "manufacturer_status": "documented",
                    "modeling_backend": "transformers"
                    if table_index
                    else ("native" if runtime == "vllm" else "pytorch"),
                    "builder_scope": "text_generation_candidate" if in_scope else "not_covered",
                    "builder_test_status": "not_tested",
                }
            )
    if len({row["architecture"] for row in result}) != len(result):
        raise ValueError("Duplicate architecture in upstream document")
    return sorted(result, key=lambda row: row["architecture"])
