# Generate a repository on Colab CPU

[Open in Colab](https://colab.research.google.com/github/ai-lab-tech/triton-hf-builder/blob/main/examples/colab/generation_only.ipynb)

Select a CPU runtime and run the cells in order: install the packages, download
a pinned SmolLM2 checkpoint, inspect it and generate a Triton model repository.
The last cell displays the configuration and validation status.
Set `triton_version` and `runtime` to choose the destination image; the CPU
example uses `triton_version="26.06"` and `runtime="python"`.

Allow about 1 GB of disk space for dependencies, the checkpoint and its copy.
Each generation creates a new output directory so the cell can be rerun.

This example does not load the model or run inference. For actual CPU serving,
use the [Docker quickstart](../docker/README.md).
