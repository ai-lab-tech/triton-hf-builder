"""Build relocatable Triton artifacts and a deployment-neutral serving manifest."""

import ast
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement

from .inspector import TASKS, inspect_model, read_json

RUNTIMES = ("transformers", "vllm", "tensorrt-llm")
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


@dataclass(frozen=True)
class BuildResult:
    repository: Path
    model_directory: Path
    manifest: dict

    def to_dict(self):
        return {
            "repository": str(self.repository),
            "model_directory": str(self.model_directory),
            "manifest": self.manifest,
        }


def _requirements(runtime, source, extras):
    # Do not copy the generator environment or pip-install engine replacements into
    # NVIDIA's specialized images. Dependency locking belongs to target-image validation.
    defaults = (
        ["numpy", "torch", "transformers>=4.51,<5", "safetensors"]
        if runtime == "transformers"
        else []
    )
    if (source / "tokenizer.model").is_file() or (source / "spiece.model").is_file():
        defaults.append("sentencepiece")
    requirements = {}
    for value in [*defaults, *extras]:
        if not isinstance(value, str) or any(c in value for c in ("\n", "\r")):
            raise ValueError("Each requirement must be one package specification")
        try:
            parsed = Requirement(value)
        except InvalidRequirement as error:
            raise ValueError(f"Invalid requirement: {value}") from error
        if parsed.url:
            raise ValueError("Use package version specifications, not URL requirements")
        key = re.sub(r"[-_.]+", "-", parsed.name).lower()
        if runtime != "transformers" and key in {
            "torch",
            "vllm",
            "tensorrt-llm",
            "tensorrt",
            "transformers",
            "numpy",
        }:
            raise ValueError(f"{key} must come from the selected {runtime} image")
        requirements[key] = str(parsed)
    return list(requirements.values())


def _copy_assets(source, destination, weights):
    # Follow file symlinks used by HF snapshots, producing a self-contained export.
    # Exclude cache directories, credentials, arbitrary source code and unrelated files.
    allowed_suffixes = {".json", ".model", ".txt", ".tiktoken", ".jinja", ".jinja2"}
    for item in source.iterdir():
        if item.is_file() and item.suffix in allowed_suffixes and not item.name.startswith("."):
            shutil.copy2(item, destination / item.name)
    for relative in weights:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    chat_templates = source / "chat_templates"
    if chat_templates.is_dir() and not chat_templates.is_symlink():
        for item in chat_templates.glob("*.jinja*"):
            if item.is_file():
                target = destination / "chat_templates" / item.name
                target.parent.mkdir(exist_ok=True)
                shutil.copy2(item, target)


def _config(name, runtime, device):
    if runtime == "vllm":
        return (
            f'name: "{name}"\nbackend: "vllm"\ninstance_group [{{ count: 1 kind: KIND_MODEL }}]\n'
        )
    # LLM API manages GPUs itself. A CPU-scheduled Python instance avoids starting
    # one engine per GPU; the deployment must still allocate gpu_count GPUs.
    kind = (
        "KIND_CPU" if runtime == "tensorrt-llm" else ("KIND_GPU" if device == "gpu" else "KIND_CPU")
    )
    return (
        f'name: "{name}"\nbackend: "python"\nmax_batch_size: 0\n'
        'input [{ name: "text_input" data_type: TYPE_STRING dims: [1] }]\n'
        'output [{ name: "text_output" data_type: TYPE_STRING dims: [1] }]\n'
        f"instance_group [{{ count: 1 kind: {kind} }}]\n"
    )


def generate_repository(
    source: str | Path,
    output: str | Path,
    *,
    name: str,
    runtime: str,
    image: str,
    task: str | None = None,
    device: str = "gpu",
    requirements: tuple[str, ...] | list[str] = (),
    tensor_parallel_size: int = 1,
    max_new_tokens: int = 128,
    max_model_len: int | None = None,
) -> BuildResult:
    """Copy a local checkpoint and generate a new repository.

    ``output`` must not exist. ``image`` identifies the target serving image; the
    builder does static checks only and does not pull images or run model inference.
    """
    if not _MODEL_NAME.fullmatch(name):
        raise ValueError("Model name must contain only letters, digits, underscores or hyphens")
    if runtime not in RUNTIMES:
        raise ValueError(f"Runtime must be one of {RUNTIMES}")
    if device not in ("cpu", "gpu"):
        raise ValueError("Device must be cpu or gpu")
    if not isinstance(image, str) or not image.strip() or any(c.isspace() for c in image):
        raise ValueError("A target serving image is required")
    image_leaf = image.rsplit("/", 1)[-1]
    if ":" not in image_leaf or image_leaf.endswith(":latest"):
        raise ValueError("Use an explicit serving image tag or digest, not latest")
    if (
        tensor_parallel_size < 1
        or max_new_tokens < 1
        or (max_model_len is not None and max_model_len < 1)
    ):
        raise ValueError("Parallel size and token limits must be positive")
    if runtime == "transformers" and tensor_parallel_size != 1:
        raise ValueError("Transformers adapter supports one device per instance")
    if runtime != "vllm" and max_model_len is not None:
        raise ValueError("max_model_len is currently a vLLM-only option")
    if runtime == "vllm" and max_new_tokens != 128:
        raise ValueError("For vLLM, set max_tokens in the inference request")
    if runtime != "transformers" and device != "gpu":
        raise ValueError(f"The {runtime} adapter requires GPU serving")
    info = inspect_model(source)
    task = task or info.inferred_task
    if task not in TASKS:
        raise ValueError(f"Specify a supported task: {', '.join(TASKS)}")
    if runtime != "transformers" and task != "text-generation":
        raise ValueError(f"The {runtime} adapter currently supports text-generation only")
    config = read_json(info.source / "config.json")
    if any(key in config for key in ("vision_config", "audio_config")):
        raise ValueError("Multimodal checkpoints need a modality-specific adapter")
    if not info.tokenizer_present:
        raise ValueError("Source must include tokenizer vocabulary files for offline serving")
    if info.inferred_task and task != info.inferred_task and task != "feature-extraction":
        raise ValueError(f"Task conflicts with checkpoint architecture ({info.inferred_task})")
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f"Output already exists: {output}")
    if output.resolve().is_relative_to(info.source):
        raise ValueError("Output must be outside the source model directory")
    deps = _requirements(runtime, info.source, requirements)
    warnings = [
        *info.warnings,
        "Runtime architecture, quantization and hardware compatibility require serving validation",
        "Requirements are not resolved or locked against the target image",
    ]
    if runtime != "transformers":
        warnings.append(
            "Use a serving image containing the selected engine and its Triton integration"
        )
    manifest = {
        "schema_version": 1,
        "generator": "triton-hf-builder",
        "generator_version": "0.1.0",
        "model_name": name,
        "runtime": runtime,
        "task": task,
        "image": image,
        "device": device,
        "gpu_count": tensor_parallel_size if device == "gpu" else 0,
        "tensor_parallel_size": tensor_parallel_size,
        "requirements_file": "requirements.txt",
        "requirements": deps,
        "validation": {"static": "passed", "serving": "not_run", "dependencies": "unlocked"},
        "model": {key: value for key, value in info.to_dict().items() if key != "source"},
        "warnings": warnings,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Build in a sibling temporary directory; failed copies never leave a repository
    # that looks deployable. No existing model repository is overwritten.
    with tempfile.TemporaryDirectory(prefix=".triton-hf-", dir=output.parent) as temporary:
        staging = Path(temporary) / "repository"
        version_dir = staging / name / "1"
        assets = version_dir / "model"
        assets.mkdir(parents=True)
        _copy_assets(info.source, assets, info.weight_files)
        (staging / name / "config.pbtxt").write_text(_config(name, runtime, device))
        settings = {
            "task": task,
            "max_new_tokens": max_new_tokens,
            "tensor_parallel_size": tensor_parallel_size,
        }
        if runtime == "vllm":
            settings = {
                "model": "./model",
                "tokenizer": "./model",
                "tensor_parallel_size": tensor_parallel_size,
                "trust_remote_code": False,
            }
            if max_model_len is not None:
                settings["max_model_len"] = max_model_len
            (version_dir / "model.json").write_text(json.dumps(settings, indent=2) + "\n")
            prepare = files("triton_hf_builder").joinpath("templates", "prepare_repository.py")
            (staging / "prepare_repository.py").write_text(prepare.read_text())
        else:
            template = "transformers_model.py" if runtime == "transformers" else "trtllm_model.py"
            model_code = files("triton_hf_builder").joinpath("templates", template).read_text()
            ast.parse(model_code)
            (version_dir / "model.py").write_text(model_code)
            (version_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
        (staging / "requirements.txt").write_text(
            "# Extra serving dependencies; resolve and pin against the target image.\n"
            + "\n".join(deps)
            + "\n"
        )
        (staging / "serving.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (staging / "README.md").write_text(_serving_readme(manifest))
        # rename fails if another process has already created a nonempty destination.
        if output.exists() or output.is_symlink():
            raise ValueError(f"Output already exists: {output}")
        staging.rename(output)
    return BuildResult(output, output / name, manifest)


def _serving_readme(manifest):
    return f"""# {manifest["model_name"]}

Generated by triton-hf-builder. Runtime: `{manifest["runtime"]}`;
task: `{manifest["task"]}`; target image: `{manifest["image"]}`.

This directory is the model repository. Upload it using triton-deploy or mount it
at your Triton server's model repository path. `serving.json` records image,
GPU count, dependencies and validation status.

Install `requirements.txt` into the serving Python environment before loading
the model. Triton does not install this file automatically. Specialized engine
images provide their own engine/CUDA/framework stack; retain those versions.
The builder package is not required in the serving image.

Static metadata checks passed. Inference and dependency resolution were not run.
Check architecture, tokenizer, quantization, runtime versions and GPU memory in
the target image. Requirements are starter specifications, not a resolved lock.
Source Transformers version: {manifest["model"].get("transformers_version") or "unknown"}.

Python adapters accept one UTF-8 string in `text_input`, shape `[1]`, and return
one UTF-8 string in `text_output`, shape `[1]`. Generation returns completion
text; classification and feature extraction return JSON in that string.
Feature extraction returns raw token features, not pooled sentence embeddings.
Generation uses raw prompts: apply your chat template on the client.
The Python adapters are synchronous, one request at a time, without streaming.
vLLM uses its native Triton generate API; its request contract differs from the
Python adapters. Set sampling limits in the request for vLLM.

HF assets are local copies, including dereferenced snapshot file symlinks.
Python custom model code is not copied or trusted. CPU is available for the
Transformers adapter; engine adapters require GPUs. For TensorRT-LLM the LLM API
selects from the container's visible GPUs; allocate exactly the requested GPUs.

For native vLLM, `model.json` initially contains portable relative paths.
triton-control's synchronized deployment resolves these to its serving mount.
For standalone serving, run `python3 /models/prepare_repository.py` inside the
container after mounting this repository at `/models`, before starting Triton.
This resolves the paths using the container's actual mount location. Repeat
preparation when moving the repository; the builder package is not required.
"""
