"""Offline regression tests for manufacturer data; no GPU or network required."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compatibility.catalog import BEGIN, END, build_catalog, change_summary, update_readme
from compatibility.parsing import image_rows, model_rows
from compatibility.sources import GitHubSources, MissingSource

IMAGE_MATRIX = """# Release Compatibility Matrix
## Container Name: vllm-python-py3
| Triton release version | NGC Tag | vLLM version | CUDA version | CUDA Driver version | Python version |
| --- | --- | --- | --- | --- | --- |
| 26.07 | nvcr.io/nvidia/tritonserver:26.07-vllm-python-py3 | 0.24.0+vendor | 13.3 | 610 | 3.12 |
| 26.06 | nvcr.io/nvidia/tritonserver:26.06-vllm-python-py3 | 0.22.1+vendor | 13.3 | 610 | 3.12 |
| 26.05 | nvcr.io/nvidia/tritonserver:26.05-vllm-python-py3 | 0.19.0 | 13.2 | 595 | 3.12 |
## Container Name: trtllm-python-py3
| Triton release version | NGC Tag | TensorRT-LLM version | CUDA version | CUDA Driver version | Python version |
| --- | --- | --- | --- | --- | --- |
| 26.07 | nvcr.io/nvidia/tritonserver:26.07-trtllm-python-py3 | 1.2.1 | 13.1 | 590 | 3.12 |
| 26.06 | nvcr.io/nvidia/tritonserver:26.06-trtllm-python-py3 | 1.2.1 | 13.1 | 590 | 3.12 |
"""
VLLM_MODELS = """# Supported Models
```python
# A comment is not a Markdown heading
```
## List of Text-only Language Models
### Generative Models
#### Text Generation
| Architecture | Models |
| --- | --- |
| `ExampleForCausalLM`, `OtherForCausalLM` | Example |
Some models are supported only via the [Transformers modeling backend](#transformers).
| Architecture | Models |
| --- | --- |
| `FallbackForCausalLM` | Fallback |
## List of Multimodal Language Models
#### Text Generation
| Architecture | Models |
| --- | --- |
| `VisionForCausalLM` | Vision |
"""
TRT_MODELS = """# Supported Models
The following is a table of supported models for the PyTorch backend:
| Architecture | Model |
| --- | --- |
| `ExampleForCausalLM` [^1] | Example |
| `VisionForConditionalGeneration` | Vision |
| `ExampleForSequenceClassification` | Classifier |
## Model-Feature Support Matrix
| Model Architecture/Feature | CUDA Graph |
| --- | --- |
| `ExampleForCausalLM` | Yes [^1] |
[^1]: Requires specific hardware.
"""
POLICY = {"recent_per_runtime": 1, "retain_releases": ["26.06"]}
SOURCE = {
    "url": "https://github.com/example/source/blob/commit/doc.md",
    "ref": "test",
    "commit": "a" * 40,
    "sha256": "b" * 64,
}


class FakeSources:
    def __init__(self, missing=False):
        self.calls = []
        self.missing = missing

    def document(self, repository, ref, path):
        self.calls.append((repository, ref, path))
        if repository == "triton-inference-server/server":
            return IMAGE_MATRIX, SOURCE
        if self.missing:
            raise MissingSource(path)
        return (VLLM_MODELS if repository == "vllm-project/vllm" else TRT_MODELS), SOURCE


class CompatibilityTests(unittest.TestCase):
    def test_summary_reports_added_and_removed_architectures(self):
        # Arrange
        current = build_catalog(FakeSources(), POLICY)
        previous = json.loads(json.dumps(current))
        previous["images"][0]["architectures"][0]["architecture"] = "OldForCausalLM"
        # Act
        summary = change_summary(previous, current)
        # Assert
        self.assertIn("+1 / -1", summary)
        self.assertIn("no serving tests were run", summary)

    def test_selects_latest_and_retained_images(self):
        # Arrange
        sources = FakeSources()
        # Act
        catalog = build_catalog(sources, POLICY)
        # Assert
        self.assertEqual(len(catalog["images"]), 4)
        self.assertEqual({r["release"] for r in catalog["images"]}, {"26.06", "26.07"})
        self.assertEqual(catalog["images"][0]["version_match"], "upstream_base_only")
        self.assertTrue(all(r["builder_test_status"] == "not_tested" for r in catalog["images"]))
        self.assertTrue(
            all(
                ref.startswith("v")
                for repo, ref, _ in sources.calls
                if repo != "triton-inference-server/server"
            )
        )

    def test_missing_exact_release_stays_unknown(self):
        # Arrange
        sources = FakeSources(missing=True)
        # Act
        catalog = build_catalog(sources, POLICY)
        # Assert
        self.assertTrue(all(r["architecture_status"] == "unknown" for r in catalog["images"]))
        self.assertFalse(
            any(
                ref in ("main", "latest")
                for repo, ref, _ in sources.calls
                if repo != "triton-inference-server/server"
            )
        )

    def test_vllm_excludes_multimodal_and_retains_modeling_backend(self):
        # Arrange
        document = VLLM_MODELS
        # Act
        rows = model_rows(document, "vllm")
        # Assert
        self.assertEqual(
            {r["architecture"] for r in rows},
            {"ExampleForCausalLM", "OtherForCausalLM", "FallbackForCausalLM"},
        )
        self.assertEqual(
            next(r for r in rows if r["architecture"] == "FallbackForCausalLM")["modeling_backend"],
            "transformers",
        )

    def test_trt_marks_non_generation_architectures_outside_scope(self):
        # Arrange
        document = TRT_MODELS
        # Act
        rows = model_rows(document, "tensorrt-llm")
        # Assert
        self.assertEqual(
            [r["architecture"] for r in rows if r["builder_scope"] == "text_generation_candidate"],
            ["ExampleForCausalLM"],
        )
        self.assertTrue(all(r["modeling_backend"] == "pytorch" for r in rows))

    def test_changed_model_document_fails_instead_of_inventing_support(self):
        # Arrange
        document = TRT_MODELS.replace("for the PyTorch backend", "for another backend")
        # Act
        with self.assertRaises(ValueError) as error:
            model_rows(document, "tensorrt-llm")
        # Assert
        self.assertIn("PyTorch backend", str(error.exception))

    def test_changed_image_table_fails(self):
        # Arrange
        document = IMAGE_MATRIX.replace("vLLM version", "Unknown version")
        # Act
        with self.assertRaises(ValueError) as error:
            image_rows(document)
        # Assert
        self.assertIn("image matrix", str(error.exception))

    def test_readme_render_is_idempotent_and_preserves_surrounding_text(self):
        # Arrange
        catalog = build_catalog(FakeSources(), POLICY)
        readme = f"Before\n{BEGIN}\nold\n{END}\nAfter"
        # Act
        rendered = update_readme(readme, catalog)
        repeated = update_readme(rendered, catalog)
        # Assert
        self.assertEqual(rendered, repeated)
        self.assertTrue(rendered.startswith("Before\n"))
        self.assertTrue(rendered.endswith("\nAfter"))
        self.assertIn("NVIDIA build differs", rendered)

    def test_missing_marker_fails_without_replacing_document(self):
        # Arrange
        catalog = build_catalog(FakeSources(), POLICY)
        # Act
        with self.assertRaises(ValueError) as error:
            update_readme("No marker", catalog)
        # Assert
        self.assertIn("marker pair", str(error.exception))

    def test_sources_pin_download_to_commit_and_cache_document(self):
        # Arrange
        sources = GitHubSources()
        sha = "a" * 40
        with patch.object(
            sources, "_read", side_effect=[json.dumps({"sha": sha}).encode(), b"# Models"]
        ) as read:
            # Act
            first = sources.document("vllm-project/vllm", "v1.0.0", "docs/models.md")
            second = sources.document("vllm-project/vllm", "v1.0.0", "docs/models.md")
        # Assert
        self.assertEqual(first, second)
        self.assertEqual(read.call_count, 2)
        self.assertIn(f"/{sha}/", read.call_args_list[1].args[0])
        self.assertEqual(first[1]["commit"], sha)

    def test_unapproved_source_is_rejected(self):
        # Arrange
        sources = GitHubSources()
        # Act
        with self.assertRaises(ValueError) as error:
            sources.document("unknown/repository", "main", "file.md")
        # Assert
        self.assertIn("Unapproved", str(error.exception))
