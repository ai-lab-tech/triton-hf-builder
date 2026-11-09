"""Destination selection stays independent of the environment running the builder."""

import json
import subprocess
import sys
from unittest.mock import patch

from test_builder import ModelFixture

from triton_hf_builder import generate_repository


class ServingSelectionTests(ModelFixture):
    def test_release_and_runtime_select_matching_image_and_backend(self):
        # Arrange
        cases = [
            ("python", "transformers", "py3", "python"),
            ("transformers", "transformers", "py3", "python"),
            ("vllm", "vllm", "vllm-python-py3", "vllm"),
            ("tensorrt-llm", "tensorrt-llm", "trtllm-python-py3", "python"),
        ]
        for runtime, canonical, suffix, backend in cases:
            with self.subTest(runtime=runtime):
                # Arrange
                output = self.root / runtime
                expected_image = f"nvcr.io/nvidia/tritonserver:26.06-{suffix}"
                # Act
                result = generate_repository(
                    self.source,
                    output,
                    name="demo",
                    runtime=runtime,
                    triton_version="26.06",
                )
                # Assert
                self.assertEqual(result.manifest["image"], expected_image)
                self.assertEqual(result.manifest["triton_version"], "26.06")
                self.assertEqual(result.manifest["runtime"], canonical)
                self.assertIn(
                    f'backend: "{backend}"', (result.model_directory / "config.pbtxt").read_text()
                )
                self.assertEqual(
                    json.loads((result.repository / "serving.json").read_text())["image"],
                    result.manifest["image"],
                )

    def test_python_alias_selects_cpu_dependency_profile_offline(self):
        # Arrange
        with patch("socket.socket", side_effect=AssertionError("Unexpected network call")):
            # Act
            result = generate_repository(
                self.source,
                self.root / "repository",
                name="demo",
                runtime="python",
                triton_version="26.06",
                device="cpu",
            )
        # Assert
        self.assertEqual(result.manifest["dependency_profile"]["runtime"], "transformers")
        self.assertIn("torch==2.9.0+cpu", result.manifest["requirements"])

    def test_custom_image_is_preserved_without_guessing_a_release(self):
        # Arrange
        image = "custom/serving:my-build"
        # Act
        result = generate_repository(
            self.source,
            self.root / "repository",
            name="demo",
            runtime="python",
            image=image,
        )
        # Assert
        self.assertEqual(result.manifest["image"], image)
        self.assertIsNone(result.manifest["triton_version"])
        self.assertIsNone(result.manifest["dependency_profile"])

    def test_missing_destination_is_rejected_without_publishing(self):
        # Arrange
        output = self.root / "repository"
        # Act
        with self.assertRaises(ValueError) as error:
            generate_repository(self.source, output, name="demo", runtime="python")
        # Assert
        self.assertIn("exactly one", str(error.exception))
        self.assertFalse(output.exists())

    def test_conflicting_destinations_are_rejected_without_publishing(self):
        # Arrange
        output = self.root / "repository"
        # Act
        with self.assertRaises(ValueError) as error:
            generate_repository(
                self.source,
                output,
                name="demo",
                runtime="python",
                triton_version="26.06",
                image="custom:1",
            )
        # Assert
        self.assertIn("exactly one", str(error.exception))
        self.assertFalse(output.exists())

    def test_invalid_releases_are_rejected_without_publishing(self):
        # Arrange
        versions = ["latest", "26.6", "26.13", "26.00", 26.06, "26.06-py3"]
        for index, version in enumerate(versions):
            with self.subTest(version=version):
                # Arrange
                output = self.root / f"repository-{index}"
                # Act
                with self.assertRaises(ValueError) as error:
                    generate_repository(
                        self.source, output, name="demo", runtime="python", triton_version=version
                    )
                # Assert
                self.assertIn("YY.MM", str(error.exception))
                self.assertFalse(output.exists())

    def test_unknown_release_generates_offline_with_profile_warning(self):
        # Arrange
        with patch("socket.socket", side_effect=AssertionError("Unexpected network call")):
            # Act
            result = generate_repository(
                self.source,
                self.root / "repository",
                name="demo",
                runtime="vllm",
                triton_version="99.01",
            )
        # Assert
        self.assertEqual(
            result.manifest["image"], "nvcr.io/nvidia/tritonserver:99.01-vllm-python-py3"
        )
        self.assertIsNone(result.manifest["dependency_profile"])
        self.assertTrue(
            any("No bundled dependency profile" in w for w in result.manifest["warnings"])
        )

    def test_cli_accepts_release_and_python_runtime(self):
        # Arrange
        arguments = [
            sys.executable,
            "-m",
            "triton_hf_builder",
            "build",
            str(self.source),
            "--output",
            str(self.root / "repository"),
            "--name",
            "demo",
            "--runtime",
            "python",
            "--triton-version",
            "26.06",
            "--device",
            "cpu",
        ]
        # Act
        completed = subprocess.run(arguments, text=True, capture_output=True, timeout=30)
        # Assert
        self.assertEqual(completed.returncode, 0, completed.stderr)
        manifest = json.loads(completed.stdout)["manifest"]
        self.assertEqual(manifest["image"], "nvcr.io/nvidia/tritonserver:26.06-py3")
        self.assertIn("torch==2.9.0+cpu", manifest["requirements"])

    def test_cli_rejects_conflicting_destination_flags(self):
        # Arrange
        output = self.root / "repository"
        arguments = [
            sys.executable,
            "-m",
            "triton_hf_builder",
            "build",
            str(self.source),
            "--output",
            str(output),
            "--name",
            "demo",
            "--runtime",
            "vllm",
            "--triton-version",
            "26.06",
            "--image",
            "custom:1",
        ]
        # Act
        completed = subprocess.run(arguments, text=True, capture_output=True, timeout=30)
        # Assert
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("not allowed with argument", completed.stderr)
        self.assertFalse(output.exists())
