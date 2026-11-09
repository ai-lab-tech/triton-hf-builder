"""Check wheel selection and generated package sources."""

from unittest import TestCase

from packaging.requirements import Requirement
from test_builder import ModelFixture

from triton_hf_builder.requirements import serving_torch_requirements


class WheelPolicyTests(TestCase):
    def test_cpu_public_pin_gets_cpu_build_and_package_source(self):
        # Arrange
        requirements = ['torch==2.9.0; python_version >= "3.10"', "numpy==1.26.4"]
        # Act
        resolved, sources = serving_torch_requirements(requirements, "cpu")
        # Assert
        self.assertEqual(resolved, [requirements[0].replace("2.9.0", "2.9.0+cpu"), requirements[1]])
        self.assertEqual(sources, ["https://download.pytorch.org/whl/cpu/torch/"])

    def test_unknown_cpu_version_requires_explicit_pin(self):
        # Arrange
        requirements = ["torch"]
        # Act
        with self.assertRaises(ValueError) as error:
            serving_torch_requirements(requirements, "cpu")
        # Assert
        self.assertIn("exact torch version", str(error.exception))

    def test_gpu_public_pin_rejects_installed_cpu_build(self):
        # Arrange
        requirements = ["torch==2.9.0"]
        # Act
        resolved, sources = serving_torch_requirements(requirements, "gpu")
        # Assert
        spec = Requirement(resolved[0]).specifier
        self.assertNotIn("2.9.0+cpu", spec)
        self.assertIn("2.9.0+cu130", spec)
        self.assertEqual(sources, [])

    def test_explicit_cuda_build_is_preserved(self):
        # Arrange
        requirements = ["torch==2.9.0+cu130"]
        # Act
        resolved, sources = serving_torch_requirements(requirements, "gpu")
        # Assert
        self.assertEqual(resolved, requirements)
        self.assertEqual(sources, ["https://download.pytorch.org/whl/cu130/torch/"])

    def test_custom_gpu_build_is_preserved_without_guessing_source(self):
        # Arrange
        requirements = ["torch==2.9.0+gpu"]
        # Act
        resolved, sources = serving_torch_requirements(requirements, "gpu")
        # Assert
        self.assertEqual(resolved, requirements)
        self.assertEqual(sources, [])

    def test_gpu_rejects_cpu_pin(self):
        # Arrange
        requirements = ["torch==2.9.0+cpu"]
        # Act
        with self.assertRaises(ValueError) as error:
            serving_torch_requirements(requirements, "gpu")
        # Assert
        self.assertIn("GPU serving cannot use", str(error.exception))

    def test_multiple_torch_packages_get_individual_sources(self):
        # Arrange
        requirements = ["torch==2.9.0", "torchvision==0.24.0"]
        # Act
        resolved, sources = serving_torch_requirements(requirements, "cpu")
        # Assert
        self.assertEqual(resolved, ["torch==2.9.0+cpu", "torchvision==0.24.0+cpu"])
        self.assertEqual(
            sources,
            [
                "https://download.pytorch.org/whl/cpu/torch/",
                "https://download.pytorch.org/whl/cpu/torchvision/",
            ],
        )


class GeneratedSourcesTests(ModelFixture):
    def test_custom_cpu_image_requires_pin_before_creating_output(self):
        # Arrange
        output = self.root / "repository"
        # Act
        with self.assertRaises(ValueError) as error:
            self.build(device="cpu")
        # Assert
        self.assertIn("exact torch version", str(error.exception))
        self.assertFalse(output.exists())

    def test_cpu_sources_match_manifest_and_requirements_file(self):
        # Arrange
        requirements = ["torch==2.9.0"]
        # Act
        result = self.build(device="cpu", requirements=requirements)
        # Assert
        self.assertEqual(result.manifest["package_indexes"], [])
        self.assertEqual(
            result.manifest["package_find_links"], ["https://download.pytorch.org/whl/cpu/torch/"]
        )
        content = (result.repository / "requirements.txt").read_text()
        self.assertIn("--find-links https://download.pytorch.org/whl/cpu/torch/", content)
        self.assertIn("torch==2.9.0+cpu", content)
        self.assertNotIn("--extra-index-url", content)
