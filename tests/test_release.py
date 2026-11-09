import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "check_release", Path(__file__).resolve().parents[1] / "scripts/check_release.py"
)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
    def test_tag_sets_version_without_manual_version_bump(self):
        # Arrange
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "_version.py"
            path.write_text('__version__ = "0.0.0"\n')
            # Act
            version = release.write_version_from_tag(path, "v0.1.5")
            # Assert
            self.assertEqual(version, "0.1.5")
            self.assertEqual(path.read_text(), '__version__ = "0.1.5"\n')

    def test_prerelease_tag_is_preserved(self):
        # Arrange
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "_version.py"
            # Act
            version = release.write_version_from_tag(path, "v0.2.0rc1")
            # Assert
            self.assertEqual(version, "0.2.0rc1")
            self.assertEqual(path.read_text(), '__version__ = "0.2.0rc1"\n')

    def test_invalid_tag_does_not_change_version_file(self):
        # Arrange
        tags = ["0.1.5", "v", "vbad", "v0.1.5+local", "v0.1.5.dev1", "v01.1.5", "v0.1.5\n"]
        for tag in tags:
            with self.subTest(tag=tag), tempfile.TemporaryDirectory() as directory:
                # Arrange
                path = Path(directory) / "_version.py"
                original = '__version__ = "0.0.0"\n'
                path.write_text(original)
                # Act
                with self.assertRaises(ValueError) as error:
                    release.write_version_from_tag(path, tag)
                # Assert
                self.assertTrue(str(error.exception))
                self.assertEqual(path.read_text(), original)

    def test_installed_wheel_matches_tag_and_runtime(self):
        # Arrange
        with (
            patch.object(release.metadata, "version", return_value="0.1.5"),
            patch("triton_hf_builder.__version__", "0.1.5"),
        ):
            # Act
            version = release.check_installed_version("v0.1.5")
        # Assert
        self.assertEqual(version, "0.1.5")

    def test_installed_wheel_with_wrong_tag_is_rejected(self):
        # Arrange
        with (
            patch.object(release.metadata, "version", return_value="0.1.2"),
            patch("triton_hf_builder.__version__", "0.1.2"),
        ):
            # Act
            with self.assertRaises(ValueError) as error:
                release.check_installed_version("v0.1.5")
        # Assert
        self.assertIn("Release tag must be v0.1.2", str(error.exception))

    def test_runtime_version_must_match_installed_metadata(self):
        # Arrange
        with (
            patch.object(release.metadata, "version", return_value="0.1.5"),
            patch("triton_hf_builder.__version__", "0.0.0"),
        ):
            # Act
            with self.assertRaises(ValueError) as error:
                release.check_installed_version("v0.1.5")
        # Assert
        self.assertIn("differs from package metadata", str(error.exception))

    def test_matching_tag(self):
        # Arrange
        source = '__version__ = "0.2.0rc1"'
        # Act
        version = release.check_version(source, "v0.2.0rc1")
        # Assert
        self.assertEqual(version, "0.2.0rc1")

    def test_mismatched_tag(self):
        # Arrange
        source = '__version__ = "0.2.0"'
        # Act
        with self.assertRaises(ValueError) as error:
            release.check_version(source, "v0.1.0")
        # Assert
        self.assertIn("Release tag must be v0.2.0", str(error.exception))

    def test_nonrelease_versions(self):
        # Arrange
        for version in ("bad", "0.2.0+local", "0.2.0.dev1", "v0.2.0"):
            with self.subTest(version=version):
                # Arrange
                source = f'__version__ = "{version}"'
                # Act
                with self.assertRaises(ValueError) as error:
                    release.check_version(source)
                # Assert
                self.assertTrue(str(error.exception))
