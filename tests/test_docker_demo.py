"""Offline checks for demo data preservation and inference failure handling."""

import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from test_builder import ModelFixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_example(name):
    spec = importlib.util.spec_from_file_location(
        f"docker_demo_{name}", ROOT / f"examples/docker/{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


demo = load_example("prepare")
client = load_example("infer")


class DockerDemoTests(ModelFixture):
    def test_existing_unmanaged_repository_is_preserved(self):
        # Arrange
        data = self.root / "data"
        repository = data / "model_repository"
        repository.mkdir(parents=True)
        marker = repository / "keep.txt"
        marker.write_text("User data")
        with patch.object(demo, "download_checkpoint") as download:
            # Act
            with self.assertRaises(ValueError) as error:
                demo.prepare(data)

        # Assert
        self.assertIn("does not match", str(error.exception))
        self.assertEqual(marker.read_text(), "User data")
        download.assert_not_called()

    def test_wrong_checkpoint_revision_is_not_published(self):
        # Arrange
        data = self.root / "data"
        with (
            patch.object(demo, "download_checkpoint", return_value=(self.source, "wrong")),
            redirect_stdout(io.StringIO()),
        ):
            # Act
            with self.assertRaises(ValueError) as error:
                demo.prepare(data)

        # Assert
        self.assertIn("pinned revision", str(error.exception))
        self.assertFalse((data / "model_repository").exists())
        self.assertFalse((data / "demo-source.json").exists())

    def test_repeated_setup_reuses_pinned_repository_without_downloading(self):
        # Arrange
        data = self.root / "data"
        with (
            patch.object(
                demo, "download_checkpoint", return_value=(self.source, demo.MODEL_REVISION)
            ) as download,
            redirect_stdout(io.StringIO()),
        ):
            # Act
            first = demo.prepare(data)
            second = demo.prepare(data)

        # Assert
        self.assertEqual(first, second)
        self.assertEqual(download.call_count, 1)
        self.assertEqual(download.call_args.args[2], {demo.MODEL_ID: demo.MODEL_REVISION})
        provenance = json.loads((data / "demo-source.json").read_text())
        self.assertEqual(provenance["revision"], demo.MODEL_REVISION)
        manifest = json.loads((first / "serving.json").read_text())
        self.assertEqual(manifest["device"], "cpu")
        self.assertEqual(set(manifest["requirements"]), set(demo.REQUIREMENTS))

    def test_empty_completion_does_not_count_as_success(self):
        # Arrange
        response = {"outputs": [{"name": "text_output", "data": [""]}]}
        with patch.object(
            client, "urlopen", return_value=io.BytesIO(json.dumps(response).encode())
        ):
            # Act
            with self.assertRaises(ValueError) as error:
                client.infer("http://unused", "Example prompt")

        # Assert
        self.assertIn("empty completion", str(error.exception))
