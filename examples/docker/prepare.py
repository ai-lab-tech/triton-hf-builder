"""Download the pinned public checkpoint and generate the CPU demo repository."""

import json
import sys
from pathlib import Path

from serving_tests.checkpoints import download_checkpoint

from triton_hf_builder import generate_repository, inspect_model
from triton_hf_builder._version import __version__

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
SERVING_IMAGE = "triton-hf-builder-demo:cpu"
REQUIREMENTS = (
    "numpy==1.26.4",
    "torch==2.9.0+cpu",
    "transformers==4.57.1",
    "safetensors==0.6.2",
    "sentencepiece==0.2.1",
    "protobuf==5.29.5",
)


def prepare(data: Path) -> Path:
    repository = data / "model_repository"
    provenance = data / "demo-source.json"
    expected = {
        "repo_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "generator_version": __version__,
        "model_name": "demo",
        "runtime": "transformers",
        "device": "cpu",
        "serving_image": SERVING_IMAGE,
        "max_new_tokens": 16,
        "requirements": list(REQUIREMENTS),
    }
    if repository.exists():
        recorded = json.loads(provenance.read_text()) if provenance.is_file() else None
        if recorded != expected:
            raise ValueError(
                "Existing demo data does not match this example. Stop the demo and move "
                "examples/docker/data aside before starting a fresh example."
            )
        inspect_model(repository / "demo/1/model")
        requirements = {
            line.strip()
            for line in (repository / "requirements.txt").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        if requirements != set(REQUIREMENTS):
            raise ValueError("Existing demo requirements were changed; refusing to overwrite them")
        print(f"Reusing generated repository: {repository}", flush=True)
        return repository

    data.mkdir(parents=True, exist_ok=True)
    cache = data / "hf-cache"
    cache.mkdir(exist_ok=True)
    print(f"Downloading {MODEL_ID}@{MODEL_REVISION}", flush=True)
    source, revision = download_checkpoint(MODEL_ID, cache, {MODEL_ID: MODEL_REVISION})
    if revision != MODEL_REVISION:
        raise ValueError("The downloaded checkpoint does not match the pinned revision")
    generate_repository(
        source,
        repository,
        name="demo",
        runtime="transformers",
        task="text-generation",
        image=SERVING_IMAGE,
        device="cpu",
        max_new_tokens=16,
        requirements=REQUIREMENTS,
    )
    provenance.write_text(json.dumps(expected, indent=2) + "\n")
    print(f"Generated repository: {repository}", flush=True)
    return repository


def main() -> int:
    try:
        prepare(Path("/data"))
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
