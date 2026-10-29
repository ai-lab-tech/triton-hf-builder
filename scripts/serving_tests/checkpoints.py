from pathlib import Path

CASES = {
    "generation": {
        "repo_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "task": "text-generation",
        "prompts": ["The capital of France is"],
    },
    "classification": {
        "repo_id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "task": "text-classification",
        "prompts": ["I absolutely loved this movie.", "This movie was awful and boring."],
        "labels": ["POSITIVE", "NEGATIVE"],
    },
    "text2text": {
        "repo_id": "google/flan-t5-small",
        "task": "text2text-generation",
        "prompts": ["translate English to German: The house is wonderful."],
    },
    "features": {
        "repo_id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "task": "feature-extraction",
        "prompts": ["A small serving test."],
    },
}


def download_checkpoint(repo_id, cache, revisions):
    from huggingface_hub import HfApi, snapshot_download

    revision = revisions.get(repo_id, "main")
    info = HfApi().model_info(repo_id, revision=revision)
    revisions.setdefault(repo_id, info.sha)
    names = [item.rfilename for item in info.siblings]
    safe_weights = [name for name in names if "/" not in name and name.endswith(".safetensors")]
    weights = safe_weights or [
        name
        for name in names
        if "/" not in name and name.startswith("pytorch_model") and name.endswith(".bin")
    ]
    if not weights:
        raise ValueError(f"No supported root checkpoint weights in {repo_id}@{info.sha}")
    suffixes = {".json", ".txt", ".model", ".tiktoken", ".jinja", ".jinja2"}
    metadata = [
        name
        for name in names
        if ("/" not in name or name.startswith("chat_templates/"))
        and Path(name).suffix in suffixes
        and not (safe_weights and name.endswith(".bin.index.json"))
        and not (not safe_weights and name.endswith(".safetensors.index.json"))
    ]
    path = snapshot_download(
        repo_id, revision=info.sha, cache_dir=str(cache), allow_patterns=[*weights, *metadata]
    )
    return Path(path), info.sha
