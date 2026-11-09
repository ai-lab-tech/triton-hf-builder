"""Keep user installs and serving tests on the generated package sources."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_builder import ModelFixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from serving_tests.docker import build_serving_image

from triton_hf_builder.dependency_profiles import dependency_profile


class PackageSourcesTests(ModelFixture):
    def test_profile_sources_survive_generation(self):
        # Arrange
        profile = dependency_profile("transformers", "nvcr.io/nvidia/tritonserver:26.06-py3", "cpu")
        profile["package_indexes"] = ["https://packages.example.org/simple"]
        profile["package_find_links"].append("https://wheels.example.org/releases/")
        with patch("triton_hf_builder.builder.dependency_profile", return_value=profile):
            # Act
            result = self.build(device="cpu")
        # Assert
        self.assertEqual(result.manifest["package_indexes"], profile["package_indexes"])
        self.assertEqual(result.manifest["package_find_links"], profile["package_find_links"])
        lines = (result.repository / "requirements.txt").read_text().splitlines()
        self.assertIn("--extra-index-url https://packages.example.org/simple", lines)
        self.assertIn("--find-links https://wheels.example.org/releases/", lines)
        self.assertEqual(lines.count("--find-links https://download.pytorch.org/whl/cpu/torch/"), 1)

    def test_serving_build_uses_only_generated_source_options(self):
        # Arrange
        for device, pin in (("cpu", "2.9.0+cpu"), ("gpu", "2.9.0+cu126")):
            with self.subTest(device=device):
                # Arrange
                directory = self.root / device
                directory.mkdir()
                repository = directory / "repository"
                repository.mkdir()
                content = f"--find-links https://wheels.example.org/{device}/\ntorch=={pin}\n"
                (repository / "requirements.txt").write_text(content)
                args = SimpleNamespace(device=device, build_timeout=60)
                with patch("serving_tests.docker.command") as command:
                    # Act
                    build_serving_image(
                        args,
                        "transformers",
                        repository,
                        "test:1",
                        "base:1",
                        directory,
                        directory / "build.log",
                    )
                # Assert
                context = directory / "image-build"
                self.assertEqual((context / "requirements.txt").read_text(), content)
                dockerfile = (context / "Dockerfile").read_text()
                self.assertIn("-r /tmp/e2e-requirements.txt", dockerfile)
                self.assertNotIn("--extra-index-url", dockerfile)
                self.assertNotIn("@INDEX_OPTION@", dockerfile)
                self.assertNotIn("TORCH_INDEX_URL", dockerfile)
                self.assertNotIn("cu130", str(command.call_args))
                self.assertNotIn("TORCH_INDEX_URL", str(command.call_args))
