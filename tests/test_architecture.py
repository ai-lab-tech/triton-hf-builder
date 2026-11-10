"""Engine architecture evidence is versioned and never implies serving success."""

import json
from unittest.mock import patch

from test_builder import ModelFixture

from triton_hf_builder import generate_repository
from triton_hf_builder.architecture import check_architectures
from triton_hf_builder.dependency_profiles import architecture_profile


class ArchitectureTests(ModelFixture):
    def test_known_architecture_is_recorded_for_both_engine_images(self):
        # Arrange
        self.config["architectures"] = ["LlamaForCausalLM"]
        self.write_config()
        # Act
        with patch("socket.socket", side_effect=AssertionError("Unexpected network call")):
            results = [
                generate_repository(
                    self.source,
                    self.root / runtime,
                    name="demo",
                    runtime=runtime,
                    triton_version="26.06",
                )
                for runtime in ("vllm", "tensorrt-llm")
            ]
        # Assert
        for result in results:
            evidence = result.manifest["architecture_compatibility"]
            self.assertEqual(evidence["status"], "upstream_documented")
            self.assertEqual(evidence["image"], result.manifest["image"])
            self.assertTrue(evidence["model_source"]["url"])
            self.assertEqual(evidence["architectures"][0]["architecture"], "LlamaForCausalLM")
            self.assertEqual(result.manifest["validation"]["serving"], "not_run")
            saved = json.loads((result.repository / "serving.json").read_text())
            self.assertEqual(saved["architecture_compatibility"], evidence)

    def test_unlisted_architecture_generates_with_an_explicit_warning(self):
        # Arrange
        self.config["architectures"] = ["UnlistedForCausalLM"]
        self.write_config()
        # Act
        result = generate_repository(
            self.source,
            self.root / "repository",
            name="demo",
            runtime="vllm",
            triton_version="26.06",
        )
        # Assert
        self.assertEqual(result.manifest["architecture_compatibility"]["status"], "not_documented")
        self.assertTrue(
            any("UnlistedForCausalLM is not listed" in w for w in result.manifest["warnings"])
        )
        self.assertTrue(result.repository.is_dir())

    def test_custom_and_digest_images_do_not_inherit_tag_documentation(self):
        # Arrange
        images = [
            "custom/vllm:26.06",
            "nvcr.io/nvidia/tritonserver:26.06-vllm-python-py3@sha256:" + "a" * 64,
            "nvcr.io/nvidia/tritonserver:99.01-vllm-python-py3",
        ]
        # Act
        results = [check_architectures("vllm", image, ("LlamaForCausalLM",)) for image in images]
        # Assert
        for result in results:
            self.assertEqual(result["status"], "unknown")
            self.assertIn("no bundled documentation", result["warnings"][0])
            self.assertNotIn("model_source", result)

    def test_missing_model_architectures_have_unknown_support(self):
        # Arrange
        image = "nvcr.io/nvidia/tritonserver:26.06-vllm-python-py3"
        # Act
        result = check_architectures("vllm", image, ())
        # Assert
        self.assertEqual(result["status"], "unknown")
        self.assertIn("no architectures", result["warnings"][0])

    def test_missing_versioned_documentation_does_not_imply_unsupported(self):
        # Arrange
        image = "nvcr.io/nvidia/tritonserver:26.06-vllm-python-py3"
        profile = {**architecture_profile("vllm", image), "architecture_status": "unknown"}
        # Act
        with patch("triton_hf_builder.architecture.architecture_profile", return_value=profile):
            result = check_architectures("vllm", image, ("LlamaForCausalLM",))
        # Assert
        self.assertEqual(result["status"], "unknown")
        self.assertIn("documentation is unavailable", result["warnings"][0])

    def test_same_architecture_is_compared_against_selected_engine_version(self):
        # Arrange
        image = "nvcr.io/nvidia/tritonserver:26.06-vllm-python-py3"
        older = {**architecture_profile("vllm", image), "architectures": []}
        # Act
        with patch("triton_hf_builder.architecture.architecture_profile", return_value=older):
            old_result = check_architectures("vllm", image, ("LlamaForCausalLM",))
        new_result = check_architectures(
            "vllm", "nvcr.io/nvidia/tritonserver:26.09-vllm-python-py3", ("LlamaForCausalLM",)
        )
        # Assert
        self.assertEqual(old_result["status"], "not_documented")
        self.assertEqual(new_result["status"], "upstream_documented")
        self.assertNotEqual(old_result["engine_version"], new_result["engine_version"])

    def test_upstream_base_version_match_preserves_limitation(self):
        # Arrange
        image = "nvcr.io/nvidia/tritonserver:26.06-vllm-python-py3"
        # Act
        result = check_architectures("vllm", image, ("LlamaForCausalLM",))
        # Assert
        self.assertEqual(result["version_match"], "upstream_base_only")
        self.assertTrue(any("not the exact NVIDIA engine build" in w for w in result["warnings"]))

    def test_transformers_modeling_backend_is_reported(self):
        # Arrange
        image = "nvcr.io/nvidia/tritonserver:26.06-vllm-python-py3"
        profile = architecture_profile("vllm", image)
        row = {**profile["architectures"][0], "modeling_backend": "transformers"}
        profile = {**profile, "architectures": [row]}
        # Act
        with patch("triton_hf_builder.architecture.architecture_profile", return_value=profile):
            result = check_architectures("vllm", image, (row["architecture"],))
        # Assert
        self.assertEqual(result["status"], "upstream_documented")
        self.assertTrue(any("Transformers modeling backend" in w for w in result["warnings"]))

    def test_outside_builder_scope_is_distinct_from_not_documented(self):
        # Arrange
        image = "nvcr.io/nvidia/tritonserver:26.06-vllm-python-py3"
        profile = architecture_profile("vllm", image)
        row = {**profile["architectures"][0], "builder_scope": "not_covered"}
        profile = {**profile, "architectures": [row]}
        # Act
        with patch("triton_hf_builder.architecture.architecture_profile", return_value=profile):
            result = check_architectures("vllm", image, (row["architecture"],))
        # Assert
        self.assertEqual(result["status"], "outside_builder_scope")
        self.assertTrue(any("outside" in w for w in result["warnings"]))

    def test_python_backend_does_not_use_engine_architecture_catalog(self):
        # Arrange
        image = "nvcr.io/nvidia/tritonserver:26.06-py3"
        # Act
        result = check_architectures("transformers", image, ("LlamaForCausalLM",))
        # Assert
        self.assertEqual(result["status"], "not_checked")
        self.assertEqual(result["warnings"], [])
