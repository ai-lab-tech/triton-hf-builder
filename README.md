# triton-hf-builder

[![CI](https://github.com/ai-lab-tech/triton-hf-builder/actions/workflows/tests.yml/badge.svg)](https://github.com/ai-lab-tech/triton-hf-builder/actions/workflows/tests.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![Status: Beta](https://img.shields.io/badge/Status-Beta-blue.svg)](#adapters-and-compatibility)

**Turn a downloaded Hugging Face checkpoint into a Triton model repository.**

The builder generates Triton configuration, runtime scripts and dependency
files for Transformers, vLLM and TensorRT-LLM. Use it through the Python API,
the `triton-hf` CLI, or the code-server extension in
[Triton Control](https://github.com/ai-lab-tech/triton-control).

[Python API](docs/python-api.md) · [Serving guide](docs/serving.md) ·
[Serving tests](scripts/README.md) · [Contributing](CONTRIBUTING.md)

## Install

Requires Python 3.10+.

```bash
python -m pip install triton-hf-builder
```

For development from a source checkout, see
[Contributing](https://github.com/ai-lab-tech/triton-hf-builder/blob/main/CONTRIBUTING.md).

## Try it with Docker

Run a complete CPU serving example from a repository checkout:

```bash
./examples/docker/quickstart.sh
```

The launcher downloads a small pinned checkpoint, generates its model repository,
builds the serving image, waits for readiness and sends an inference request.
No GPU or local Python installation is needed. The first run downloads large
images and dependencies; later runs reuse cached files and build layers.
See the [Docker quickstart](https://github.com/ai-lab-tech/triton-hf-builder/blob/main/examples/docker/README.md)
for requirements, requests and cleanup.

## Quick start

Point `source` to a complete local checkpoint with weights and tokenizer files.
Choose a new `output` directory outside the checkpoint directory.

Generation copies the checkpoint assets without loading weights and runs without
a GPU. Allow disk space for a complete checkpoint copy.

```python
from triton_hf_builder import generate_repository

result = generate_repository(
    source="/workspace/hf-checkpoint",
    output="/workspace/model_repository",
    name="my_model",
    runtime="transformers",
    image="nvcr.io/nvidia/tritonserver:26.06-py3",
    device="cpu",
)
print(result.model_directory / "config.pbtxt")
```

**No manual `requirements` needed:** the builder writes runtime defaults into
`requirements.txt`. Optional overrides, all parameters and return values are
explained in the [Python API](docs/python-api.md).

The equivalent CLI:

```bash
triton-hf inspect /workspace/hf-checkpoint
triton-hf build /workspace/hf-checkpoint \
  --output /workspace/model_repository --name my_model \
  --runtime transformers --device cpu \
  --image nvcr.io/nvidia/tritonserver:26.06-py3
```

## Generated output

For the Transformers example above, a checkpoint with a single Safetensors weight
file produces the following layout. Checkpoint filenames vary with the source.

```text
model_repository/
├── my_model/
│   ├── config.pbtxt                 # Triton model configuration
│   └── 1/
│       ├── model.py                 # Triton Python adapter
│       ├── settings.json            # Task and generation settings
│       └── model/
│           ├── config.json
│           ├── model.safetensors    # Copied checkpoint weights
│           ├── tokenizer.json
│           ├── README.md            # Original model card, if supplied
│           └── LICENSE              # Original model license, if supplied
├── requirements.txt                # Serving dependencies
├── serving.json                    # Runtime, image and build metadata
├── LICENSE.triton-hf-builder        # License for generated support scripts
└── README.md                       # Serving instructions
```

TensorRT-LLM uses the same layout with its own Python adapter. For vLLM,
`my_model/1/model.json` replaces `model.py` and `settings.json`, and
`prepare_repository.py` is added at the repository root.

Deploy the `model_repository/` root to Triton after preparing its serving
dependencies; generation alone does not validate inference.
See the [serving guide](docs/serving.md) for deployment and vLLM path preparation.

## Adapters and compatibility

| Builder adapter (`runtime`) | Triton backend | Model execution | Tasks | Device |
| --- | --- | --- | --- | --- |
| `transformers` | Python | Hugging Face Transformers with PyTorch | Generation, text-to-text, classification, raw features | CPU or GPU |
| `vllm` | vLLM | vLLM engine | Text generation | GPU |
| `tensorrt-llm` | Python | TensorRT-LLM LLM API with PyTorch backend | Text generation | GPU |

The API parameter `runtime` selects the builder adapter. The Triton backend and
model execution framework are separate layers.

Recognized multimodal checkpoints and custom model Python are outside the current
adapter scope. See [runtime limitations](docs/serving.md#runtimes-and-request-contracts).

### Manufacturer compatibility catalog

<!-- compatibility:begin -->

Manufacturer-documented text-generation candidates. [Full catalog and architecture names](compatibility.json).

| Image tag | Engine version | CUDA / reported driver | Architectures |
| --- | --- | --- | --- |
| `26.09-vllm-python-py3` | `0.29.0+3b480f30.nv26.9.68265578` | 13.4.1.012 / 615.71.09 | [118 candidates](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/docs/models/supported_models.md) † |
| `26.06-vllm-python-py3` | `0.22.1+7b9cb5b7.nv26.6.55098374` | 13.3.0.035 / 610.43.02 | [129 candidates](https://github.com/vllm-project/vllm/blob/0decac0d96c42b49572498019f0a0e3600f50398/docs/models/supported_models.md) † |
| `26.07-trtllm-python-py3` | `1.2.1` | 13.1.0.036 / 590.44.01 | [18 candidates](https://github.com/NVIDIA/TensorRT-LLM/blob/376f7e1bd8ed543f75014309e3fd4b237e9b0e73/docs/source/models/supported-models.md) |
| `26.06-trtllm-python-py3` | `1.2.1` | 13.1.0.036 / 590.44.01 | [18 candidates](https://github.com/NVIDIA/TensorRT-LLM/blob/376f7e1bd8ed543f75014309e3fd4b237e9b0e73/docs/source/models/supported-models.md) |

Source: [NVIDIA image matrix](https://github.com/triton-inference-server/server/blob/eb74dea5d400b92f54eacbd5f3f84903c786e4cd/docs/introduction/compatibility.md). † Upstream base version; NVIDIA build differs.

These are documentation references, not serving guarantees. Check the linked matrices for quantization, GPU and feature restrictions; the reported driver is not a universal minimum.

<!-- compatibility:end -->

The catalog tracks 26.06 and the latest documented image per runtime. A weekly
workflow proposes updates as PRs. [Maintenance and selection policy](scripts/README.md#compatibility-catalog-maintenance).

## Development and releases

See [CONTRIBUTING.md](CONTRIBUTING.md) for local checks and
[RELEASING.md](RELEASING.md) for package publication and code-server integration.
Real GPU serving tests are opt-in; follow the [test guide](scripts/README.md).

[Apache-2.0 license](LICENSE) · [Licensing and test models](docs/licensing.md) ·
[Security policy](SECURITY.md) · [Code of Conduct](CODE_OF_CONDUCT.md)
Model weights and serving engines retain their own licenses.
