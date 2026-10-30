# CPU serving quickstart

Download a small public checkpoint, generate its Triton model repository, and
send a real inference request. The example uses the Transformers adapter and
[SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
at commit `12fd25f77366fa6b3b4b768ec3050bf629380bac`.

## Run

Use a source checkout of this repository with Docker Engine or Docker Desktop,
BuildKit, and Docker Compose v2.20 or newer. The serving image targets Linux
x86_64 containers with Python 3.12. No local Python installation or GPU is needed.

From the repository root:

```bash
./examples/docker/quickstart.sh
```

The launcher:

1. Builds a small setup image containing the checked-out builder.
2. Downloads the pinned checkpoint and generates a CPU model named `demo`.
3. Builds the serving image using that repository's exact `requirements.txt`.
4. Starts Triton and waits for both server and model readiness.
5. Sends an inference request and prints its prompt and nonempty completion.

The first run downloads large images and Python wheels. The Triton base image
alone occupies about 23 GB unpacked; allow additional space for image-build
caches and both the cached and exported checkpoint. Later runs reuse the model
cache, generated repository and Docker build layers.

The HTTP endpoint is bound to `127.0.0.1:8000`. To choose another host port:

```bash
DEMO_PORT=18080 ./examples/docker/quickstart.sh
```

## Send another request

From the repository root, using the Python client already inside the container:

```bash
docker compose -f examples/docker/compose.yaml exec -T triton \
  python3 /opt/demo/infer.py --prompt "Once upon a time"
```

Or use the Triton HTTP API from the host:

```bash
curl --fail-with-body http://127.0.0.1:8000/v2/models/demo/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[{"name":"text_input","shape":[1],"datatype":"BYTES","data":["The capital of France is"]}],"outputs":[{"name":"text_output"}]}'
```

The model generates at most 16 new tokens per request. This is a small
text-generation example; it is not a chat interface or a performance benchmark.

## Files and dependencies

Generated assets stay outside the images:

```text
examples/docker/
├── data/
│   ├── hf-cache/              # Download cache, including model-card attribution
│   ├── model_repository/      # Generated Triton repository, mounted read-only
│   └── demo-source.json       # Model revision and generation settings
└── requirements.generated.txt # Exact generated serving dependency file
```

Downloaded and generated assets are excluded from Git and the source
distribution. Docker build contexts include only the required source files and
`requirements.generated.txt`, never the weights or checkpoint cache.
No model weights are included in either image. The builder is installed only in
the setup image. The server's dependencies are installed during its image build
into a separate Python environment and are not reinstalled on startup.

The demo pins PyTorch `2.9.0+cpu`, Transformers `4.57.1`, NumPy `1.26.4`,
Safetensors `0.6.2`, SentencePiece `0.2.1` and Protobuf `5.29.5`. PyTorch comes
from its CPU wheel index. The serving base is
`nvcr.io/nvidia/tritonserver:26.06-py3`; vLLM and TensorRT-LLM use different serving
stacks and are outside this CPU example.

The model card declares Apache-2.0 at the pinned revision. Model-card and license
documents are preserved with the exported assets. NVIDIA's container and its
components retain their own terms. See [Licensing](../../docs/licensing.md).

## Logs and cleanup

```bash
docker compose -f examples/docker/compose.yaml logs --tail 100 triton
docker compose -f examples/docker/compose.yaml down
```

Stopping the demo preserves downloaded files and images. To start from fresh
data, stop the server, move `examples/docker/data` aside, and run the launcher
again. The setup refuses to overwrite an unrelated repository or reuse data from
a different checkpoint revision or demo configuration.

If setup or a build fails, fix the reported error and rerun the launcher. If
readiness fails, the launcher prints the server logs. Existing services on port
8000 can be avoided with `DEMO_PORT`.

## Validation

The default example was exercised on Linux x86_64 CPU on 2026-10-30 with Docker
Engine 29.9.0 and Compose 5.6.0. The run downloaded the pinned checkpoint,
generated the repository, built the images, passed server and model readiness,
and returned nonempty completions through both the container client and the
published host port. A second launcher run reused the repository and build
layers successfully. This validates the pinned CPU example, not other models or
GPU runtimes.
