"""Offline regression tests for manufacturer data; no GPU or network required."""

import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compatibility.catalog import BEGIN, END, build_catalog, change_summary, update_readme
from compatibility.dependencies import build_dependency_catalog
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
REVIEWED_PROFILE = {
    "runtime": "vllm",
    "image": "nvcr.io/nvidia/tritonserver:26.06-vllm-python-py3",
    "device": "gpu",
    "status": "manufacturer_documented",
    "requirements": ["sentencepiece==0.2.1"],
    "package_indexes": [],
    "sources": [{"url": "https://example.com/reviewed-rule"}],
    "notes": ["Reviewed tokenizer dependency"],
}
IMAGE_CORRECTION = {
    "runtime": "tensorrt-llm",
    "image": "nvcr.io/nvidia/tritonserver:26.06-trtllm-python-py3",
    "requirements": ["openai==2.8.1"],
    "sources": [SOURCE],
    "notes": ["Import correction only"],
}


def curated_rules(*, profiles=(), corrections=()):
    return {
        "schema_version": 1,
        "profiles": copy.deepcopy(list(profiles)),
        "image_corrections": copy.deepcopy(list(corrections)),
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
    def test_dependency_catalog_preserves_image_variants_and_scoped_corrections(self):
        # Arrange
        catalog = build_catalog(FakeSources(), POLICY)
        curated = curated_rules(corrections=[IMAGE_CORRECTION])
        # Act
        bundled = build_dependency_catalog(catalog, curated)
        # Assert
        self.assertEqual(len(bundled["profiles"]), 4)
        corrected = [p for p in bundled["profiles"] if p["requirements"]]
        self.assertEqual(len(corrected), 1)
        self.assertEqual(corrected[0]["image"], curated["image_corrections"][0]["image"])
        self.assertTrue(all(p["status"] == "manufacturer_documented" for p in bundled["profiles"]))

    def test_dependency_catalog_leaves_inputs_unchanged(self):
        # Arrange
        catalog = build_catalog(FakeSources(), POLICY)
        curated = curated_rules(profiles=[REVIEWED_PROFILE], corrections=[IMAGE_CORRECTION])
        original_catalog = copy.deepcopy(catalog)
        original_curated = copy.deepcopy(curated)
        # Act
        build_dependency_catalog(catalog, curated)
        # Assert
        self.assertEqual(catalog, original_catalog)
        self.assertEqual(curated, original_curated)

    def test_dependency_catalog_bundles_versioned_architecture_evidence(self):
        # Arrange
        catalog = build_catalog(FakeSources(), POLICY)
        # Act
        result = build_dependency_catalog(catalog, curated_rules())
        # Assert
        self.assertEqual(len(result["architecture_profiles"]), len(catalog["images"]))
        for profile, row in zip(result["architecture_profiles"], catalog["images"]):
            self.assertEqual(profile["architectures"], row["architectures"])
            self.assertEqual(profile["model_source"], row["model_source"])
            self.assertEqual(profile["version_match"], row["version_match"])

    def test_unknown_engine_mapping_can_be_bundled_without_version_match(self):
        # Arrange
        catalog = build_catalog(FakeSources(), POLICY)
        row = catalog["images"][0]
        row.pop("version_match")
        row.update(architecture_status="unknown", architectures=[], model_source=None)
        # Act
        result = build_dependency_catalog(catalog, curated_rules())
        # Assert
        self.assertIsNone(result["architecture_profiles"][0]["version_match"])
        self.assertEqual(result["architecture_profiles"][0]["architecture_status"], "unknown")

    def test_reviewed_profile_extends_existing_engine_without_losing_source_facts(self):
        # Arrange
        catalog = build_catalog(FakeSources(), POLICY)
        curated = curated_rules(profiles=[REVIEWED_PROFILE])
        # Act
        result = build_dependency_catalog(catalog, curated)
        # Assert
        self.assertEqual(len(result["profiles"]), 4)
        profile = next(
            p for p in result["profiles"] if p["image"] == curated["profiles"][0]["image"]
        )
        self.assertEqual(profile["requirements"], ["sentencepiece==0.2.1"])
        self.assertEqual(profile["environment"]["engine_version"], "0.22.1+vendor")
        self.assertEqual(profile["sources"], [SOURCE, *curated["profiles"][0]["sources"]])

    def test_invalid_reviewed_rules_fail_before_publication(self):
        # Arrange
        catalog = build_catalog(FakeSources(), POLICY)
        cases = [
            (
                "package_indexes",
                ["https://example.com\n--trusted-host example.com"],
                "single HTTPS",
            ),
            ("requirements", ["torch==2.9.0\nother==1"], "single-line"),
            ("requirements", ["torch @ https://example.com/torch.whl"], "URLs"),
            ("package_find_links", ["file:///tmp/wheels"], "HTTPS"),
            ("package_find_links", ["https://user:password@example.org/"], "credentials"),
            ("package_find_links", "https://example.org", "list"),
            ("status", "verified", "status"),
            ("sources", [], "source URLs"),
        ]
        for field, value, message in cases:
            with self.subTest(field=field, value=value):
                # Arrange
                curated = curated_rules(profiles=[REVIEWED_PROFILE])
                curated["profiles"][0][field] = value
                # Act
                with self.assertRaises(ValueError) as error:
                    build_dependency_catalog(catalog, curated)
                # Assert
                self.assertIn(message, str(error.exception))

    def test_duplicate_reviewed_rules_are_rejected(self):
        # Arrange
        catalog = build_catalog(FakeSources(), POLICY)
        for field in ("profiles", "image_corrections"):
            with self.subTest(field=field):
                # Arrange
                curated = curated_rules(profiles=[REVIEWED_PROFILE], corrections=[IMAGE_CORRECTION])
                curated[field].append(copy.deepcopy(curated[field][0]))
                # Act
                with self.assertRaises(ValueError) as error:
                    build_dependency_catalog(catalog, curated)
                # Assert
                self.assertIn("Duplicate", str(error.exception))

    def test_reviewed_rule_cannot_overwrite_manufacturer_environment(self):
        # Arrange
        catalog = build_catalog(FakeSources(), POLICY)
        curated = curated_rules(profiles=[REVIEWED_PROFILE])
        curated["profiles"][0]["environment"] = {"cuda_version": "wrong"}
        # Act
        with self.assertRaises(ValueError) as error:
            build_dependency_catalog(catalog, curated)
        # Assert
        self.assertIn("disagrees with manufacturer", str(error.exception))

    def test_bundled_dependency_catalog_matches_maintenance_inputs(self):
        # Arrange
        root = Path(__file__).resolve().parents[1]
        catalog = json.loads((root / "compatibility.json").read_text())
        curated = json.loads((root / "dependency-profiles.json").read_text())
        bundled = json.loads((root / "src/triton_hf_builder/dependency_profiles.json").read_text())
        # Act
        expected = build_dependency_catalog(catalog, curated)
        # Assert
        self.assertEqual(bundled, expected)

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
