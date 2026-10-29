# Python API

[Quick start](../README.md#quick-start) · [Serving guide](serving.md)


Import the public API from `triton_hf_builder`. No code-server, Docker daemon or
inference framework is needed to inspect a checkpoint or generate its repository.

### `generate_repository(...) -> BuildResult`

Reads a local checkpoint, validates metadata and options, and copies the required
assets into a new Triton model repository. Generation does not start a server,
install dependencies or run inference. Arguments after `source` and `output`
must be passed by keyword.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `source: str \| Path` | Required | Local HF checkpoint directory with `config.json`, supported weights and tokenizer files. Use a downloaded directory, not a Hugging Face model ID or URL. |
| `output: str \| Path` | Required | New repository root. Must not exist and must be outside `source`. Missing parent directories are created. |
| `name: str` | Required | Triton model name and subdirectory name. Start with a letter or digit; then use letters, digits, `_` or `-`. |
| `runtime: str` | Required | `transformers`, `vllm` or `tensorrt-llm`. See the task support table below. |
| `image: str` | Required | Target serving image with an explicit tag or SHA-256 digest. `latest` is rejected. This is recorded in the manifest; the builder does not pull or inspect it. |
| `task: str \| None` | `None` | Infer from checkpoint metadata. If ambiguous, choose `text-generation`, `text2text-generation`, `text-classification` or `feature-extraction`. Engine runtimes accept only `text-generation`. |
| `device: str` | `"gpu"` | Serving device: `cpu` or `gpu`. CPU is supported only by `transformers`. This does not select a GPU index or require a GPU on the build machine. |
| `requirements: Sequence[str]` | `()` | Optional serving package overrides or additions, for example `["transformers==4.57.1"]`. Omit this to use automatic defaults. An empty sequence still includes defaults. |
| `tensor_parallel_size: int` | `1` | Number of GPUs used together by vLLM or TensorRT-LLM. Must remain `1` for Transformers. GPU allocation is performed by your deployment. |
| `max_new_tokens: int` | `128` | Generated-token limit for Python generation adapters. Does not affect classification/features. For native vLLM, leave this unchanged and set `max_tokens` in the inference request. |
| `max_model_len: int \| None` | `None` | vLLM-only context limit (prompt plus output tokens). Omit to leave it to the engine. Other runtimes reject this option. |

Integer options must be positive integers; booleans and fractional values are
rejected. An explicit `task` must agree with recognized architecture metadata,
except that `feature-extraction` may be selected explicitly for Transformers.

The returned `BuildResult` contains:

| Attribute | Type | Meaning |
| --- | --- | --- |
| `repository` | `Path` | Absolute repository root, corresponding to `output`. Deploy or mount this directory. |
| `model_directory` | `Path` | Model subdirectory, `repository / name`, containing `config.pbtxt` and version `1`. |
| `manifest` | `dict` | Metadata also saved as `serving.json`: runtime, image, task, GPU count, requirements, warnings and validation status. |
| `to_dict()` | `dict` | Result with path fields converted to strings, suitable for `json.dumps`. |

```python
import json

print(result.repository / "requirements.txt")
print(result.manifest["requirements"])
print(result.manifest["validation"])
# {"static": "passed", "serving": "not_run", "dependencies": "unlocked"}
print(json.dumps(result.to_dict(), indent=2))
```

Success means static checks and artifact generation completed. It does not mean
that the selected image can load the model. Read `manifest["warnings"]` and test
serving in the target environment. Modifying the returned dictionary does not
rewrite `serving.json` or regenerate files.

### `inspect_model(source: str | Path) -> ModelInfo`

Inspect a downloaded checkpoint before choosing generation options:

```python
from triton_hf_builder import inspect_model

info = inspect_model("/workspace/hf-checkpoint")
print(info.inferred_task)       # A supported task name, or None
print(info.tokenizer_present)   # Whether recognized tokenizer assets exist
print(info.warnings)
```

Inspection reads metadata and checks weight-file presence and shard references;
it does not load or validate tensor contents. It writes no repository. A checkpoint
can pass inspection but fail generation, for example when its tokenizer is missing
or multimodal metadata is recognized.

| `ModelInfo` attribute | Meaning |
| --- | --- |
| `source` | Resolved local directory as a `Path`. |
| `model_type`, `architectures` | Model type and tuple of architecture names from metadata. |
| `inferred_task` | Inferred task, or `None` when explicit selection is needed. |
| `weight_files` | Sorted tuple of checkpoint filenames relative to `source`, including indexed shards. |
| `tokenizer_present` | Whether recognized tokenizer vocabulary files exist. |
| `custom_code` | Whether model/tokenizer metadata declares custom code; generated adapters still disable it. |
| `multimodal` | Whether recognized vision/audio configuration is present. |
| `quantization` | Declared quantization method, or `None`; no compatibility validation is performed. |
| `transformers_version` | Version recorded by the checkpoint, if present. It is informational and does not set dependencies automatically. |
| `warnings` | Tuple of inspection warnings. |
| `to_dict()` | Metadata dictionary with `source` converted to a string. |

### Errors and recovery

Both functions raise `ValueError` for invalid metadata or unsupported inputs,
such as missing weights, unsafe shard references or an ambiguous task during
generation. Generation also rejects existing output paths, unsupported runtime
options and invalid requirement declarations. Filesystem failures such as denied
permissions or insufficient disk space propagate as `OSError` subclasses.

```python
try:
    info = inspect_model("/workspace/hf-checkpoint")
except ValueError as error:
    print(f"Check the checkpoint metadata: {error}")
except OSError as error:
    print(f"Cannot access the checkpoint: {error}")
```

Generation builds in a temporary sibling directory and publishes the output
after copying completes. Normal exceptions clean up the temporary build and
leave existing output untouched. Allow disk space for the full copied checkpoint;
a forced process termination can leave a `.triton-hf-*` temporary directory.

## Serving dependencies

**Start by omitting `requirements`.** You do not need to know package versions
to generate your first repository. Dependencies come from the selected runtime's
built-in defaults, with a small tokenizer-based addition:

| Runtime | Automatically generated requirements |
| --- | --- |
| `transformers` | `numpy`, `torch`, `transformers>=4.51,<5`, `safetensors` |
| `vllm` / `tensorrt-llm` | No engine packages: use the engine stack already supplied by the specialized image. The requirements file can be empty apart from comments. |
| Any runtime with `tokenizer.model` or `spiece.model` | Adds `sentencepiece`. |

The builder always writes `requirements.txt`. You can inspect it before deploying:

```python
print((result.repository / "requirements.txt").read_text())
```

### Advanced: override versions or add packages

Use `requirements` when you already know a compatible version or a model-specific
extra dependency. For example, pass `requirements=["transformers==4.57.1"]` to a
Transformers build to replace its default Transformers constraint while retaining
the other defaults. This version is an override example, not a universal recommendation.
The CLI equivalent is `--requirement 'transformers==4.57.1'`; repeat the option
for additional packages.

Pass a list or tuple of individual package specifications, not a path to a file
or a single string. Unconditional declarations for the same normalized package
name replace earlier ones. Conditional declarations such as
`'some-package==1.0; sys_platform == "linux"'` are retained alongside defaults;
they do not remove mandatory packages on other platforms. URL requirements,
multiline declarations and pip command-line options are rejected.

For engine runtimes, overrides of `torch`, `vllm`, `tensorrt-llm`, `tensorrt`,
`transformers` and `numpy` are rejected to preserve the specialized image's stack.
Other extra packages may be supplied, but their compatibility still needs testing.

### What automatic defaults do and do not guarantee

`requirements.txt` contains **serving dependencies**, never a freeze of the
machine running the generator. Transformers builds start with `numpy`, `torch`,
`transformers>=4.51,<5`, and `safetensors`. This initial pipeline adapter targets
Transformers 4.x. SentencePiece vocabularies add `sentencepiece`.
Supply version overrides or extra packages with repeated `--requirement` options.
Specialized vLLM / TensorRT-LLM images provide their engine and framework packages;
the builder rejects overrides for core engine packages in these modes.

Requirements are starter specifications, not a lock. The generator does not
inspect or pull images, resolve dependencies, infer every optional model package,
or certify CUDA compatibility. Resolve and pin packages in the target image,
preserving its CUDA/framework stack. The Python version used by a custom
execution environment must match Triton's Python backend stub.

Triton itself does not install `requirements.txt`. Standalone deployments must
install it before model loading or supply a compatible Python execution
environment. `triton-control` already accepts `requirements_txt` and installs
those packages before starting the server. Its editor integration reads the
generated image, GPU count and requirements into the deployment form for review.
No Dockerfile is required for that workflow. The builder is not needed at runtime.

`serving.json` is deployment-neutral metadata with `schema_version: 1`.
The CLI prints JSON to stdout on success and JSON errors to stderr on build or
inspection failures. Integrations should check the exit code. Static checks are
reported separately from serving validation, which remains `not_run` until you
test the model in the target environment.

