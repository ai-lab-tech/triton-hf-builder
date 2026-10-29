# Real Hugging Face → Triton serving tests

This opt-in runner downloads actual checkpoints, invokes the builder CLI, starts
Docker Triton, waits for readiness and sends real HTTP inference requests. It
never imports model frameworks on the host. No inference results are claimed
until you execute it successfully.

## Setup and execution

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[e2e]'

# All four Transformers tasks on CPU
python scripts/test_serving.py --runtime transformers --device cpu

# All three runtimes, sequentially, on one GPU
python scripts/test_serving.py --runtime all --device gpu --gpu-device 0
```

Docker must be accessible to your user. GPU tests need an NVIDIA driver and
NVIDIA Container Toolkit. Missing images and checkpoints are downloaded. Images
are large; allow space in Docker storage and in the work directory for both
HF cache and copied checkpoints. The host does not need PyTorch.

Individual cases/runtimes:

```bash
python scripts/test_serving.py --runtime transformers --device gpu --case classification
python scripts/test_serving.py --runtime vllm --device gpu
python scripts/test_serving.py --runtime tensorrt-llm --device gpu
```

## Cases and assertions

| Case | Checkpoint | Checks |
| --- | --- | --- |
| Generation | `HuggingFaceTB/SmolLM2-135M-Instruct` | Nonempty completion; Transformers parity with direct inference |
| Classification | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | Positive/negative labels, confidence > 0.9 and direct inference parity |
| Text-to-text | `google/flan-t5-small` | Nonempty translation and direct inference parity |
| Features | Same DistilBERT checkpoint | Raw token features match direct inference within numerical tolerance |

Transformers runs all four cases by default; engine runtimes run generation only.
Every case starts its own real server on an automatically allocated localhost-only
port. Cases run sequentially. Direct Transformers reference inference runs inside
the container from the same local weights independently of generated `model.py`.
This temporarily loads a second small model; allow memory for that.

Engine completions must be nonempty; cross-engine numerical parity is not asserted.
Python adapters must also reject an invalid tensor shape with a client error.
An artifact generation pass or a successful health check alone cannot pass a case.

## Images

Default Triton release: `26.06`. Transformers builds a **test-only image** from
`nvcr.io/nvidia/tritonserver:26.06-py3`, installing generated `requirements.txt`.
Its test dependencies are PyTorch 2.9.0, Transformers 4.57.1, NumPy 1.26.4,
Safetensors 0.6.2, SentencePiece 0.2.1 and Protobuf 5.29.5. The PyTorch wheel index
is CPU for CPU tests and CUDA 13.0 for GPU tests.

These defaults are test inputs, not verified compatibility guarantees. The first
real run may reveal image, dependency or GPU incompatibility, reported as failures.
Change `--torch-version` or `--transformers-version` as needed. Use an existing
prepared image with `--transformers-image registry/image:tag`; it must already
contain the dependencies for the selected checkpoints.

vLLM and TensorRT-LLM use official `26.06-vllm-python-py3` and
`26.06-trtllm-python-py3` images, retaining their engine/framework stacks.
Override with `--vllm-image`, `--trtllm-image`, `--python-base-image` or
`--triton-version`. Existing local images are reused. The generated vLLM
preparation script resolves checkpoint paths inside the serving mount.

## Artifacts and reproduction

```text
.e2e-artifacts/
├── hf-cache/                         # Reused checkpoints
└── runs/<timestamp-and-id>/
    ├── report.json                   # Overall pass/fail status
    ├── preflight.log
    └── <runtime>-<case>/
        ├── repository/               # Generated Triton model repository
        ├── result.json               # Commit, image ID, versions, outcome
        ├── commands.log
        ├── triton.log
        ├── model-metadata.json
        ├── inference-0.json           # Request, response, reference output
        ├── invalid-request.json      # Python adapters
        └── failure.txt               # On failure
```

Use `--work-dir /path/to/artifacts` to choose another disk. Each run gets a new
directory and preserves earlier results. Successful cases mark their copied
repository's `serving.json` validation as passed and link the evidence.
Resolved model commit SHAs, image IDs/digests and installed framework versions
are recorded. Reproduce models with repeated revision options:

```bash
python scripts/test_serving.py --runtime transformers --device cpu \
  --case generation \
  --revision HuggingFaceTB/SmolLM2-135M-Instruct=COMMIT_FROM_RESULT
```

Exit status: 0 only when all selected cases pass; 1 for failures; 130 for
interruption. The runner removes only its uniquely named test containers after
saving logs, including failures. `--keep-containers` retains them for debugging.
Images, caches and artifacts remain available. Startup timeout defaults to 900
seconds, inference to 300 seconds, and image build/pull to 3600 seconds. Each has
a corresponding CLI option; see `python scripts/test_serving.py --help`.
