import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "check_release", Path(__file__).resolve().parents[1] / "scripts/check_release.py"
)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
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
                source = f'__version__ = "{version}"'
                # Act
                with self.assertRaises(ValueError) as error:
                    release.check_version(source)
                # Assert
                self.assertTrue(str(error.exception))
