# triton-hf-builder

Build a Triton model repository from an **already downloaded Hugging Face
checkpoint**. Use the Python library or CLI independently, or invoke the CLI
from the code-server extension in `triton-control`.

Generation reads JSON metadata and copies assets. It does not import PyTorch,
Transformers, vLLM or TensorRT-LLM, load weights, download models, or use a GPU.
Disk space is required for a full copy of the checkpoint.

**Status: alpha.** Static generation support is distinct from validated serving
compatibility. See [contributing](CONTRIBUTING.md), [releases](RELEASING.md) and
[security](SECURITY.md). Project code is licensed under [Apache-2.0](LICENSE);
downloaded models and serving engines retain their own licenses.

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/ai-lab-tech/triton-hf-builder.git
cd triton-hf-builder
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
triton-hf inspect /workspace/hf-checkpoint
```

## Build

Choose an explicit target image containing the selected backend. The image below
matches the existing triton-control examples; choose your deployment's release.

```bash
triton-hf build /workspace/hf-checkpoint \
  --output /workspace/model_repository \
  --name my_model \
  --runtime vllm \
  --image nvcr.io/nvidia/tritonserver:26.06-vllm-python-py3 \
  --max-model-len 8192
```

The output directory must not exist. This prevents accidental replacement of
other models. HF snapshot file symlinks are dereferenced; the exported files
remain usable after moving the repository to another machine or object store.

```text
model_repository/
├── my_model/
│   ├── config.pbtxt
│   └── 1/
│       ├── model/                 # Local checkpoint and tokenizer assets
│       ├── model.json             # vLLM only
│       ├── model.py               # Python adapters only
│       └── settings.json          # Python adapters only
├── requirements.txt
├── serving.json
├── prepare_repository.py           # vLLM only; standalone mount path preparation
└── README.md
```

Python usage:

```python
from triton_hf_builder import generate_repository

result = generate_repository(
    source="/workspace/hf-checkpoint",
    output="/workspace/model_repository",
    name="my_model",
    runtime="transformers",
    image="nvcr.io/nvidia/tritonserver:26.06-py3",
    device="cpu",
    requirements=["transformers==4.57.1"],
)
print(result.model_directory / "config.pbtxt")
```

The version specification demonstrates an override; select versions that support
your checkpoint and target image. It is not a compatibility guarantee.

## Adapters and compatibility

| Runtime | Tasks | Generated adapter |
| --- | --- | --- |
| `transformers` | Text generation, text-to-text generation, classification, feature extraction | Triton Python backend, Hugging Face pipeline |
| `vllm` | Text generation | Native Triton vLLM backend and `model.json` |
| `tensorrt-llm` | Text generation | Triton Python backend using public `LLM` API with `backend="pytorch"` |

Adapters operate across compatible HF architectures rather than a single model
name. This is not a universal support matrix: runtime release, GPU hardware,
quantization and architecture still determine whether a checkpoint can load.
Multimodal checkpoints are rejected when recognized. Image/audio tasks, custom
Python model code, adapter-only checkpoints and legacy compiled TensorRT engines
need additional adapters and are outside this initial implementation.

Task inference recognizes common language-model and sequence-classification
architecture names. Set `--task` when metadata is ambiguous. Explicit tasks are
recorded choices, not proof that the runtime supports the architecture.
Use `--task feature-extraction` for raw token features; pooling and normalization
for sentence embeddings are not implemented.

Python adapters use a unary contract: `text_input` is a UTF-8 string tensor of
shape `[1]`; `text_output` has the same shape. Classification and features are
JSON encoded in that output string. They process requests serially without
streaming or Triton dynamic batching. Text-generation outputs exclude the prompt.
Apply chat templates on the client. vLLM retains its native generate API and
sampling parameters; set `max_tokens` in its request.

Native vLLM does not resolve model paths relative to `model.json`. The generated
paths start as `./model` for portability. triton-control rewrites them to `/models`
through its synchronized repository deployment. For standalone use, run the
generated `python3 /models/prepare_repository.py` inside the serving container
after mounting the repository at `/models` and before starting Triton. It resolves
paths to the actual mount location without installing the builder. Repeat this
step after moving the repository. The repository must be writable for this step.

`--tensor-parallel-size` configures vLLM and TensorRT-LLM GPU counts. Transformers
supports one CPU/GPU per instance. TensorRT-LLM's Python process is CPU scheduled
by Triton while the LLM engine owns the deployment's visible GPUs. Give it exactly
the GPUs listed in `serving.json`; avoid multiple model engines competing for
the same GPU allocation.

## Serving dependencies

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

## Development

The Python package owns inspection and generation. `builder.py` coordinates the
build; `options.py` validates inputs; `runtimes.py` declares runtime capabilities
and renders Triton files; `requirements.py` resolves dependency declarations;
`artifacts.py` copies checkpoint assets and writes serving instructions. None of
these modules loads model weights. Engine imports belong in the generated templates.

The code-server extension calls the JSON CLI through its `hf-builder-client.js`;
the editor wizard stays in `hf-builder.js`. Real-serving test orchestration lives
in `scripts/serving_tests/`, with `scripts/test_serving.py` as the command entry point.

```bash
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

The adapter contract tests use lightweight fake engines and fake tensors; they
exercise lifecycle, paths, device selection and responses without weights or a
GPU. Real engine compatibility and GPU inference need deployment testing.

## Real serving tests

The opt-in runner downloads real Hugging Face checkpoints, invokes the builder,
starts Triton in Docker, and sends inference requests. Run it yourself:

```bash
python -m pip install -e '.[e2e]'
python scripts/test_serving.py --runtime all --device gpu
```

For Transformers on CPU, use `--runtime transformers --device cpu`.
See [the serving-test guide](scripts/README.md) for checkpoints, assertions,
image options, saved logs, reproduction and cleanup. The runner has not yet
been validated against a real serving environment.

Reference documentation:

- [Triton Python backend and execution environments](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html)
- [Triton vLLM backend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/vllm_backend/README.html)
- [TensorRT-LLM public LLM API](https://nvidia.github.io/TensorRT-LLM/llm-api/index.html)
