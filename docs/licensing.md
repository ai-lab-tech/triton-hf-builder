# Licensing

## Project code and tests

The project's Python library, CLI, generated support-code templates,
documentation, unit tests, synthetic fixtures and serving-test runner are
licensed under [Apache-2.0](../LICENSE). This does not change the licenses of
third-party dependencies, model assets or container contents.

The Docker quickstart scripts are covered by the same project license. Its
serving image includes a copy of that license for its inference client; model
weights remain outside both demo images in the local cache and exported assets.

The unit tests construct their checkpoint metadata and placeholder weight bytes
locally. They do not download or bundle real checkpoints or datasets. The source
distribution includes the tests; the wheel does not. Both distributions include
the project's complete Apache-2.0 license.

Generated model repositories include `LICENSE.triton-hf-builder` for the Python
support scripts. Those scripts also carry an Apache-2.0 SPDX identifier.
Checkpoint license, notice and model-card documents present at the checkpoint
root are preserved beside the copied model assets. These documents remain
separate from the generated repository's own README.

The builder preserves supplied documents; it does not establish ownership or
grant rights to a checkpoint. A checkpoint can have additional terms or notices
outside its root directory or its model card.

## Serving-test checkpoints

The optional serving runner downloads the following external checkpoints.
Their model cards were checked on 2026-10-30 at the revisions below. Each declares
Apache-2.0. The checkpoints and their training datasets are not included in the
builder's Git repository, wheel or source distribution.

| Checkpoint | Tests | Declared license and checked revision |
| --- | --- | --- |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | Generation | [Apache-2.0; `12fd25f`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/README.md) |
| `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | Classification, features | [Apache-2.0; `714eb0f`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/blob/714eb0fa89d2f80546fda750413ed43d93601a13/README.md) |
| `google/flan-t5-small` | Text-to-text | [Apache-2.0; `0fc9ddf`](https://huggingface.co/google/flan-t5-small/blob/0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab/README.md) |

At those revisions, all three publish their license declaration in `README.md`
without a separate root license file. The downloader now retains that model
card as well as root license and notice files when present. Each test run records
the resolved commit; use `--revision REPO_ID=COMMIT` to reproduce it. The default
branch can change, so the declarations above apply to the checked revisions.

## Python dependencies

The direct dependencies and the explicit serving-test dependency versions were
checked against their published metadata and upstream license files. These are
the upstream projects' licenses; their binary wheels can contain additional
components and notices.

| Component | Checked version | Project license |
| --- | --- | --- |
| [packaging](https://pypi.org/project/packaging/26.3/) | 26.3 | Apache-2.0 OR BSD-2-Clause |
| [protobuf](https://pypi.org/project/protobuf/6.33.5/) | 6.33.5 | BSD-3-Clause |
| [tritonclient](https://pypi.org/project/tritonclient/2.73.0/) | 2.73.0 | BSD-3-Clause |
| [huggingface-hub](https://pypi.org/project/huggingface-hub/2.4.0/) | 2.4.0 | Apache-2.0 |
| [build](https://pypi.org/project/build/1.6.1/) | 1.6.1 | MIT |
| [twine](https://pypi.org/project/twine/6.2.0/) | 6.2.0 | Apache-2.0 |
| [ruff](https://pypi.org/project/ruff/0.17.3/) | 0.17.3 | MIT |
| [PyTorch](https://pypi.org/project/torch/2.9.0/) | 2.9.0 | BSD-3-Clause |
| [Transformers](https://pypi.org/project/transformers/4.57.1/) | 4.57.1 | Apache-2.0 |
| [NumPy](https://pypi.org/project/numpy/1.26.4/) | 1.26.4 | BSD-3-Clause, plus bundled-component notices |
| [Safetensors](https://pypi.org/project/safetensors/0.6.2/) | 0.6.2 | Apache-2.0 |
| [SentencePiece](https://github.com/google/sentencepiece/blob/v0.2.1/LICENSE) | 0.2.1 | Apache-2.0 |
| [OpenAI Python SDK](https://pypi.org/project/openai/2.8.1/) | 2.8.1 | Apache-2.0 |
| [pip](https://pypi.org/project/pip/25.2/) | 25.2 | MIT |

Dependencies are installed separately and are not vendored into this package.
This review does not cover every transitive dependency or every version allowed
by the dependency ranges. For example, NumPy's published binary-license metadata
also lists native runtime components with GCC runtime exceptions and LGPL terms.
Retain and review the actual installed distributions' notices when distributing
a complete serving environment.

## Upstream catalogs and container images

The compatibility catalog records facts and links to pinned upstream documents.
Its upstream source projects have their own licenses:

- [Triton Server at `eb74dea`](https://github.com/triton-inference-server/server/blob/eb74dea5d400b92f54eacbd5f3f84903c786e4cd/LICENSE): BSD-3-Clause.
- [vLLM at `98dff2a`](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/LICENSE): Apache-2.0.
- [TensorRT-LLM at `376f7e1`](https://github.com/NVIDIA/TensorRT-LLM/blob/376f7e1bd8ed543f75014309e3fd4b237e9b0e73/LICENSE): Apache-2.0, with separate notices for included components.

The runner pulls NVIDIA containers and builds local test images. This package
does not distribute those images. Open-source licenses of Triton or TensorRT-LLM
do not establish the terms of every component in an NVIDIA container. Review
the selected image's terms and included notices before redistributing images;
see [NVIDIA's software agreements](https://www.nvidia.com/en-us/agreements/enterprise-software/).

The review covers the repository's visible attribution, synthetic fixtures,
declared direct dependencies and packaged files. It does not independently prove
the authorship of each source line or certify complete third-party containers.
