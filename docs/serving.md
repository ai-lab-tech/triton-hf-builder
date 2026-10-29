# Serving guide

[Quick start](../README.md#quick-start) · [Python API](python-api.md) · [Real serving tests](../scripts/README.md)

## Generated repository

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

Deploy the repository root. Install its generated `requirements.txt` in the serving
environment before Triton loads the model. Triton itself does not install it;
triton-control can install the requirements during deployment. The builder package
is not needed in the serving image. See [dependency defaults and overrides](python-api.md#serving-dependencies).

## Runtimes and request contracts

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

