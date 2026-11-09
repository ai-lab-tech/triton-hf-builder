"""Runtime capabilities and Triton artifact rendering. No engine imports."""

import ast
import json
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from .inspector import TASKS
from .model_config import render_model_config
from .options import BuildOptions


@dataclass(frozen=True)
class RuntimeAdapter:
    name: str
    tasks: tuple[str, ...]
    model_template: str | None
    instance_kind: str | None
    image_suffix: str
    gpu_required: bool = True
    tensor_parallel: bool = True
    max_model_len: bool = False
    default_requirements: tuple[str, ...] = ()
    managed_packages: frozenset[str] = frozenset()

    def resolve_image(self, *, image: str | None, triton_version: str | None) -> str:
        if (image is None) == (triton_version is None):
            raise ValueError("Specify exactly one of triton_version or image")
        if image is not None:
            return image
        if not isinstance(triton_version, str) or not re.fullmatch(
            r"[0-9]{2}\.(?:0[1-9]|1[0-2])", triton_version
        ):
            raise ValueError("triton_version must use YY.MM format, for example '26.06'")
        return f"nvcr.io/nvidia/tritonserver:{triton_version}-{self.image_suffix}"

    def validate(self, options: BuildOptions) -> None:
        if not self.tensor_parallel and options.tensor_parallel_size != 1:
            raise ValueError("Transformers adapter supports one device per instance")
        if not self.max_model_len and options.max_model_len is not None:
            raise ValueError("max_model_len is currently a vLLM-only option")
        if self.model_template is None and options.max_new_tokens != 128:
            raise ValueError("For vLLM, set max_tokens in the inference request")
        if self.gpu_required and options.device != "gpu":
            raise ValueError(f"The {self.name} adapter requires GPU serving")

    def write_artifacts(self, repository: Path, options: BuildOptions, task: str) -> None:
        model = repository / options.name
        version = model / "1"
        kind = self.instance_kind or ("KIND_GPU" if options.device == "gpu" else "KIND_CPU")
        backend = "python" if self.model_template else self.name
        config = render_model_config(
            options.name, backend, kind, python_io=self.model_template is not None
        )
        (model / "config.pbtxt").write_text(config, encoding="utf-8")
        if self.model_template:
            _write_template(self.model_template, version / "model.py")
            settings = {
                "task": task,
                "max_new_tokens": options.max_new_tokens,
                "tensor_parallel_size": options.tensor_parallel_size,
            }
            settings_path = version / "settings.json"
        else:
            settings = {
                "model": "./model",
                "tokenizer": "./model",
                "trust_remote_code": False,
                "tensor_parallel_size": options.tensor_parallel_size,
            }
            if options.max_model_len is not None:
                settings["max_model_len"] = options.max_model_len
            settings_path = version / "model.json"
            _write_template("prepare_repository.py", repository / "prepare_repository.py")
        settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def _write_template(name: str, target: Path) -> None:
    content = files("triton_hf_builder").joinpath("templates", name).read_text(encoding="utf-8")
    ast.parse(content)
    target.write_text(content, encoding="utf-8")


_ENGINE_PACKAGES = frozenset({"torch", "vllm", "tensorrt-llm", "tensorrt", "transformers", "numpy"})
ADAPTERS = {
    "transformers": RuntimeAdapter(
        name="transformers",
        tasks=TASKS,
        model_template="transformers_model.py",
        instance_kind=None,
        image_suffix="py3",
        gpu_required=False,
        tensor_parallel=False,
        default_requirements=("numpy", "torch", "transformers>=4.51,<5", "safetensors"),
    ),
    "vllm": RuntimeAdapter(
        name="vllm",
        tasks=("text-generation",),
        model_template=None,
        instance_kind="KIND_MODEL",
        image_suffix="vllm-python-py3",
        max_model_len=True,
        managed_packages=_ENGINE_PACKAGES,
    ),
    # CPU scheduling starts one Python process; the LLM engine owns visible GPUs.
    "tensorrt-llm": RuntimeAdapter(
        name="tensorrt-llm",
        tasks=("text-generation",),
        model_template="trtllm_model.py",
        instance_kind="KIND_CPU",
        image_suffix="trtllm-python-py3",
        managed_packages=_ENGINE_PACKAGES,
    ),
}
RUNTIMES = ("python", *ADAPTERS)


def get_adapter(runtime: str) -> RuntimeAdapter:
    if runtime == "python":
        runtime = "transformers"
    if not isinstance(runtime, str) or runtime not in ADAPTERS:
        raise ValueError(f"Runtime must be one of {RUNTIMES}")
    return ADAPTERS[runtime]
