"""Orchestrate metadata inspection, runtime rendering and repository publication."""

import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ._version import __version__
from .artifacts import copy_assets, serving_readme
from .inspector import TASKS, ModelInfo, inspect_model
from .licensing import generator_license
from .options import BuildOptions
from .requirements import resolve_requirements
from .runtimes import RuntimeAdapter, get_adapter


@dataclass(frozen=True)
class BuildResult:
    repository: Path
    model_directory: Path
    manifest: dict

    def to_dict(self) -> dict:
        return {
            "repository": str(self.repository),
            "model_directory": str(self.model_directory),
            "manifest": self.manifest,
        }


def generate_repository(
    source: str | Path,
    output: str | Path,
    *,
    name: str,
    runtime: str,
    image: str,
    task: str | None = None,
    device: str = "gpu",
    requirements: Sequence[str] = (),
    tensor_parallel_size: int = 1,
    max_new_tokens: int = 128,
    max_model_len: int | None = None,
) -> BuildResult:
    """Generate a new repository without loading weights or running inference.

    ``output`` must not exist. The target image and requirements describe the
    intended serving environment; compatibility still requires runtime testing.
    """
    options = BuildOptions(
        name, image, device, task, tensor_parallel_size, max_new_tokens, max_model_len
    )
    adapter = get_adapter(runtime)
    adapter.validate(options)
    info = inspect_model(source)
    selected_task = _select_task(info, adapter, task)
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f"Output already exists: {output}")
    if output.resolve().is_relative_to(info.source):
        raise ValueError("Output must be outside the source model directory")
    deps = resolve_requirements(adapter, info.source, requirements, image=image)
    manifest = _manifest(info, adapter, options, selected_task, deps)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".triton-hf-", dir=output.parent) as temporary:
        staging = Path(temporary) / "repository"
        assets = staging / name / "1" / "model"
        assets.mkdir(parents=True)
        copy_assets(info.source, assets, info.weight_files)
        adapter.write_artifacts(staging, options, selected_task)
        (staging / "requirements.txt").write_text(
            "# Extra serving dependencies; resolve and pin against the target image.\n"
            + "\n".join(deps)
            + "\n",
            encoding="utf-8",
        )
        (staging / "serving.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "LICENSE.triton-hf-builder").write_text(generator_license(), encoding="utf-8")
        (staging / "README.md").write_text(serving_readme(manifest), encoding="utf-8")
        if output.exists() or output.is_symlink():
            raise ValueError(f"Output already exists: {output}")
        staging.rename(output)
    return BuildResult(output, output / name, manifest)


def _select_task(info: ModelInfo, adapter: RuntimeAdapter, requested: str | None) -> str:
    task = requested if requested is not None else info.inferred_task
    if task not in TASKS:
        raise ValueError(f"Specify a supported task: {', '.join(TASKS)}")
    if task not in adapter.tasks:
        raise ValueError(
            f"The {adapter.name} adapter currently supports {', '.join(adapter.tasks)} only"
        )
    if info.multimodal:
        raise ValueError("Multimodal checkpoints need a modality-specific adapter")
    if not info.tokenizer_present:
        raise ValueError("Source must include tokenizer vocabulary files for offline serving")
    if info.inferred_task and task != info.inferred_task and task != "feature-extraction":
        raise ValueError(f"Task conflicts with checkpoint architecture ({info.inferred_task})")
    return task


def _manifest(
    info: ModelInfo,
    adapter: RuntimeAdapter,
    options: BuildOptions,
    task: str,
    requirements: list[str],
) -> dict:
    warnings = [
        *info.warnings,
        "Runtime architecture, quantization and hardware compatibility require serving validation",
        "Requirements are not resolved or locked against the target image",
    ]
    if adapter.gpu_required:
        warnings.append(
            "Use a serving image containing the selected engine and its Triton integration"
        )
    return {
        "schema_version": 1,
        "generator": "triton-hf-builder",
        "generator_version": __version__,
        "model_name": options.name,
        "runtime": adapter.name,
        "task": task,
        "image": options.image,
        "device": options.device,
        "gpu_count": options.tensor_parallel_size if options.device == "gpu" else 0,
        "tensor_parallel_size": options.tensor_parallel_size,
        "requirements_file": "requirements.txt",
        "requirements": requirements,
        "validation": {"static": "passed", "serving": "not_run", "dependencies": "unlocked"},
        "model": {key: value for key, value in info.to_dict().items() if key != "source"},
        "warnings": warnings,
    }
