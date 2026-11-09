"""Check that image verification handles profiled requirements and still fails conflicts."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from test_builder import ModelFixture

from triton_hf_builder import generate_repository

spec = importlib.util.spec_from_file_location(
    "verify_requirements", Path(__file__).resolve().parents[1] / "scripts/verify_requirements.py"
)
verification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verification)


class RequirementsVerificationTests(ModelFixture):
    def test_profile_file_with_index_verifies_installed_versions(self):
        # Arrange
        result = generate_repository(
            self.source,
            self.root / "repository",
            name="demo",
            runtime="transformers",
            image="nvcr.io/nvidia/tritonserver:26.06-py3",
            device="cpu",
        )
        installed = dict(value.split("==", 1) for value in result.manifest["requirements"])
        with patch.object(
            verification.importlib.metadata, "version", side_effect=installed.__getitem__
        ):
            # Act
            verified = verification.verify(result.repository / "requirements.txt")
        # Assert
        self.assertEqual(len(verified["packages"]), len(installed))
        self.assertEqual(len(verified["sha256"]), 64)

    def test_index_option_does_not_hide_an_incompatible_package(self):
        # Arrange
        path = self.root / "requirements.txt"
        path.write_text(
            "--extra-index-url https://download.pytorch.org/whl/cpu\ntorch==2.9.0+cpu\n"
        )
        with patch.object(verification.importlib.metadata, "version", return_value="2.8.0"):
            # Act
            with self.assertRaises(RuntimeError) as error:
                verification.verify(path)
        # Assert
        self.assertIn("installed version is 2.8.0", str(error.exception))
