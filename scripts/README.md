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

Every case follows the complete path independently:

1. Download the checkpoint and generate its Triton repository and `requirements.txt`.
2. Build a dedicated image from that runtime's base image and install **that exact file**.
3. Verify every declared package/version, then load the model in a new Triton container.
4. Send inference requests and check their responses.

A case cannot reuse another case's dependency installation. Each installation
layer gets a unique case ID, so it executes even when requirements match another
case. Base layers and downloaded wheels remain cached. Empty requirements files
for specialized engine images still go through installation and verification;
those engines use the dependencies supplied by their base image.

Engine completions must be nonempty; cross-engine numerical parity is not asserted.
Python adapters must also reject an invalid tensor shape with a client error.
An artifact generation pass or a successful health check alone cannot pass a case.

## Images

Default Triton release: `26.06`. Every case builds its own **test-only image**.
Transformers uses `nvcr.io/nvidia/tritonserver:26.06-py3` as the base.
Its test dependencies are PyTorch 2.9.0, Transformers 4.57.1, NumPy 1.26.4,
Safetensors 0.6.2, SentencePiece 0.2.1 and Protobuf 5.29.5. The PyTorch wheel index
is CPU for CPU tests and CUDA 13.0 for GPU tests. The generated requirement pins
the matching PyTorch variant (`+cpu` or `+cu130`). All model dependencies are
installed with `pip install -r requirements.txt`; there is no separate PyTorch
installation that could hide a missing requirement.

The test image installs a separate pip 25.2 under `/opt/triton-hf-builder-pip`
for downloads with a 180-second socket timeout and retries/resumption. It keeps
Debian's system pip intact. A BuildKit cache retains completed downloads across
build attempts; interrupted downloads can resume within an active pip invocation.

These defaults are test inputs, not verified compatibility guarantees. The first
real run may reveal image, dependency or GPU incompatibility, reported as failures.
Change `--torch-version` or `--transformers-version` as needed. Use an existing
base image with `--transformers-image registry/image:tag`. Each case still
installs its generated requirements on top of that base.

vLLM and TensorRT-LLM use official `26.06-vllm-python-py3` and
`26.06-trtllm-python-py3` base images, retaining their engine/framework stacks and
installing each case's generated extra requirements in a dedicated derived image.
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
        ├── e2e-install-report.json    # pip's installation report
        ├── e2e-requirements-verified.json # File hash and installed package versions
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

The derived serving image must exist locally and its identity is recorded.
BuildKit can build successfully while keeping the base image only in its cache,
without a local base-image tag. In that case `base_image` retains the requested
reference with `id: null`, empty `digests`, and `inspection_status:
not_in_local_store`; testing continues. This does not establish the exact base
digest. Other Docker inspection failures still fail the case.

Exit status: 0 only when all selected cases pass; 1 for failures; 130 for
interruption. The runner removes only its uniquely named test containers after
saving logs, including failures. `--keep-containers` retains them for debugging.
Images, caches and artifacts remain available. Startup timeout defaults to 900
seconds, inference to 300 seconds, and image build/pull to 3600 seconds. Each has
a corresponding CLI option; see `python scripts/test_serving.py --help`.

For the TensorRT-LLM 26.06 base image, the runner applies the builder's shared
`openai==2.8.1` correction when generating requirements for its derived test image.
The pin is installed and verified through the same per-case `requirements.txt`
workflow, not through an extra manual installation. See the
[image correction and sources](../docs/python-api.md#serving-dependencies).

## Compatibility catalog maintenance

The manufacturer catalog is independent of serving tests. It downloads small
official documentation files; it does not download weights, pull Docker images,
install serving engines or run inference. No AI service or API key is needed.

```bash
# Refresh the catalog and generated README block from official sources.
python scripts/update_compatibility.py

# Verify the saved JSON and README agree, without network access.
python scripts/update_compatibility.py --check
```

`compatibility-policy.json` selects how many recent images to track per runtime
and which releases to retain. Initially this is one latest documented image per
runtime plus 26.06. Updating this policy does not change builder, extension or
serving-test image defaults. "Latest documented" is not a recommendation.

The updater follows this chain:

1. NVIDIA's Triton `docs/introduction/compatibility.md` supplies the precise
   image-to-engine, Python, CUDA and reported driver mapping. The source is pinned
   to the last commit that changed that file.
2. The exact engine release tag selects the versioned model document in the
   official vLLM or TensorRT-LLM repository. Tags are resolved to immutable commits.
   A NVIDIA-specific version suffix is preserved; an upstream base-version match
   is explicitly weaker than documentation for the exact vendor build.
3. Only the vLLM text-generation sections and TensorRT-LLM PyTorch-backend model
   table are imported. Each architecture retains the modeling backend and builder
   scope. The conservative candidate filter accepts `*ForCausalLM` and
   `GPT2LMHeadModel`; other entries remain `not_covered`, not "unsupported".
4. `compatibility.json` records source URLs, refs, commit hashes and SHA-256 hashes.
   The compact README table is generated from that JSON; full architecture names remain in the catalog.

All architecture entries mean **documented upstream**, not "tested with this
builder". Quantization, GPU support, custom-code requirements and feature
restrictions remain unverified; follow the immutable model-source links for the
complete matrices and footnotes. In particular, being listed does not establish
that a checkpoint works with `trust_remote_code=False` or fits GPU memory.
The reported driver value is not reinterpreted as the minimum compatible driver.
Missing architectures can also work through upstream fallback mechanisms; the
catalog is not a denylist and does not block generation.

Missing exact-release documents remain `unknown`; there is no fallback to current
`main`/`latest` model documentation. HTTP errors other than 404 and unfamiliar
table layouts fail the refresh before tracked files are written. This makes format
changes visible instead of silently publishing an empty or guessed support list.
Unchanged upstream files yield identical output, avoiding timestamp-only PRs.

### GitHub automation

`.github/workflows/update-compatibility.yml` runs every Monday at 06:23 UTC and can
also be started with **Actions → Update manufacturer compatibility → Run workflow**.
Once the workflow is on the default branch, it updates the existing
`automation/manufacturer-compatibility` PR or opens one when changes exist. Only
`compatibility.json` and `README.md` are committed. It does not automatically merge.

Enable **Settings → Actions → General → Workflow permissions → Allow GitHub
Actions to create and approve pull requests** (the organization may control this).
The workflow needs `contents: write` and `pull-requests: write`; its built-in
`GITHUB_TOKEN` also authenticates source API reads to avoid anonymous rate limits.
The token is sent only to the GitHub API, never to raw-content downloads.

PRs created with `GITHUB_TOKEN` normally do not trigger another PR workflow, so
the updater runs unit tests, lint, formatting and offline catalog consistency
checks itself before opening the PR. If branch protection requires ordinary PR
CI, use an appropriately scoped GitHub App token for PR creation or arrange an
explicit CI run; do not bypass required checks. These GitHub settings are not
configured by adding workflow files. Scheduled runs may be delayed, and GitHub
may disable schedules on inactive public repositories; the manual trigger remains
useful for recovery.

Upstream references:

- [NVIDIA image compatibility matrix](https://github.com/triton-inference-server/server/blob/main/docs/introduction/compatibility.md)
- [vLLM model documentation](https://github.com/vllm-project/vllm/blob/main/docs/models/supported_models.md)
- [TensorRT-LLM model documentation](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/models/supported-models.md)
- [GitHub scheduled workflows](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows/#schedule)
- [Automated PR permissions and token behavior](https://github.com/peter-evans/create-pull-request#workflow-permissions)
