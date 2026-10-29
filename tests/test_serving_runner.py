"""Test runner failure handling with mocks; never download models or start Docker."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from serving_tests import cli, inference, runner, support


class ServingRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_failed_http_response_is_saved(self):
        # Arrange
        path = self.root / "inference.json"
        response = {"error": "model failed"}
        with patch.object(inference, "request_json", return_value=(500, response)):
            # Act
            with self.assertRaises(AssertionError) as error:
                inference.python_inference("http://unused", "demo", "Hello", 1, path)
        # Assert
        self.assertIn("HTTP 500", str(error.exception))
        self.assertEqual(json.loads(path.read_text())["response"], response)

    def test_interrupted_case_updates_shared_result(self):
        # Arrange
        result = {"runtime": "transformers", "case": "generation"}
        with patch.object(runner, "download_checkpoint", side_effect=KeyboardInterrupt):
            # Act
            with self.assertRaises(KeyboardInterrupt):
                runner.run_case(
                    SimpleNamespace(),
                    "transformers",
                    "generation",
                    "test:1",
                    self.root,
                    self.root,
                    {},
                    result=result,
                )
        # Assert
        self.assertEqual(result["status"], "interrupted")
        saved = json.loads((self.root / "transformers-generation/result.json").read_text())
        self.assertEqual(saved, result)

    def test_interrupted_case_remains_in_top_level_report(self):
        # Arrange
        def interrupt(*args, result):
            result["status"] = "interrupted"
            raise KeyboardInterrupt

        with (
            patch.object(cli.importlib.util, "find_spec", return_value=object()),
            patch.object(cli, "command"),
            patch.object(cli, "run_case", side_effect=interrupt),
        ):
            # Act
            code = cli.main(["--work-dir", str(self.root)])
        # Assert
        self.assertEqual(code, 130)
        report = json.loads(next(self.root.glob("runs/*/report.json")).read_text())
        self.assertEqual(report["status"], "interrupted")
        self.assertEqual(report["results"][0]["status"], "interrupted")

    def test_missing_dependency_fails_before_docker(self):
        # Arrange
        with (
            patch.object(cli.importlib.util, "find_spec", return_value=None),
            patch.object(cli, "command") as command,
        ):
            # Act
            code = cli.main(["--work-dir", str(self.root)])
        # Assert
        self.assertEqual(code, 1)
        command.assert_not_called()
        report = json.loads(next(self.root.glob("runs/*/report.json")).read_text())
        self.assertIn("huggingface_hub", report["error"])

    def test_failed_json_serialization_preserves_previous_report(self):
        # Arrange
        path = self.root / "report.json"
        path.write_text('{"status": "running"}')
        # Act
        with self.assertRaises(TypeError):
            support.write_json(path, {"invalid": object()})
        # Assert
        self.assertEqual(json.loads(path.read_text()), {"status": "running"})
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_boolean_is_not_numeric_reference_match(self):
        # Arrange
        actual, expected = [True], [1.0]
        # Act
        matches = inference.close_values(actual, expected)
        # Assert
        self.assertFalse(matches)
