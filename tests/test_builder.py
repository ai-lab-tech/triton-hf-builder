import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from triton_hf_builder import generate_repository, inspect_model


class ModelFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "hf"
        self.source.mkdir()
        self.config = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}
        self.write_config()
        (self.source / "tokenizer.json").write_text("{}")
        # These are deliberately not loadable weights. Generation must never load them.
        (self.source / "model.safetensors").write_bytes(b"fixture-weight-bytes")

    def write_config(self):
        (self.source / "config.json").write_text(json.dumps(self.config))

    def build(self, runtime="transformers", **kwargs):
        return generate_repository(
            self.source,
            self.root / "repository",
            name="demo",
            runtime=runtime,
            image="test/serving:1.0",
            **kwargs,
        )


class BuilderTests(ModelFixture):
    def test_inspection_and_build_do_not_import_frameworks(self):
        # Arrange
        script = """
import sys
from triton_hf_builder import inspect_model, generate_repository
info = inspect_model(sys.argv[1])
generate_repository(sys.argv[1], sys.argv[2], name="demo", runtime="vllm", image="test:1")
assert info.inferred_task == "text-generation"
assert not {"torch", "transformers", "vllm", "tensorrt_llm"}.intersection(sys.modules)
"""

        # Act
        result = subprocess.run(
            [sys.executable, "-c", script, str(self.source), str(self.root / "output")],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Assert
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_vllm_export_is_relocatable_and_preserves_weights(self):
        # Arrange
        original = self.root / "cached-weight"
        original.write_bytes(b"cache contents")
        (self.source / "model.safetensors").unlink()
        (self.source / "model.safetensors").symlink_to(original)

        # Act
        result = self.build("vllm", tensor_parallel_size=2, max_model_len=4096)
        original.unlink()

        # Assert
        exported = result.model_directory / "1/model/model.safetensors"
        self.assertFalse(exported.is_symlink())
        self.assertEqual(exported.read_bytes(), b"cache contents")
        settings = json.loads((result.model_directory / "1/model.json").read_text())
        self.assertEqual(settings["model"], "./model")
        self.assertEqual(settings["max_model_len"], 4096)
        self.assertEqual(result.manifest["gpu_count"], 2)
        self.assertEqual(result.manifest["validation"]["serving"], "not_run")
        self.assertFalse((result.model_directory / "1/model.py").exists())

    def test_transformers_requirements_override_and_cpu(self):
        # Arrange
        dependencies = ["transformers==4.57.1", "sentencepiece==0.2.1"]

        # Act
        result = self.build(device="cpu", requirements=dependencies)

        # Assert
        requirements = (result.repository / "requirements.txt").read_text().splitlines()
        self.assertIn("transformers==4.57.1", requirements)
        self.assertNotIn("transformers", requirements)
        self.assertIn("KIND_CPU", (result.model_directory / "config.pbtxt").read_text())
        self.assertEqual(result.manifest["gpu_count"], 0)

    def test_standalone_vllm_preparation_uses_actual_mount_path(self):
        # Arrange
        result = self.build("vllm")
        moved = self.root / "mounted-elsewhere"
        result.repository.rename(moved)

        # Act
        completed = subprocess.run(
            [sys.executable, str(moved / "prepare_repository.py")],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Assert
        self.assertEqual(completed.returncode, 0, completed.stderr)
        settings = json.loads((moved / "demo/1/model.json").read_text())
        self.assertEqual(settings["model"], str(moved / "demo/1/model"))
        self.assertEqual(settings["tokenizer"], settings["model"])

    def test_trtllm_uses_python_llm_api_not_engine_artifacts(self):
        # Arrange: use the local checkpoint from ModelFixture.setUp.

        # Act
        result = self.build("tensorrt-llm")

        # Assert
        code = (result.model_directory / "1/model.py").read_text()
        self.assertIn("from tensorrt_llm import LLM", code)
        self.assertIn('backend="pytorch"', code)
        self.assertIn("KIND_CPU", (result.model_directory / "config.pbtxt").read_text())
        self.assertEqual(result.manifest["gpu_count"], 1)
        self.assertEqual(result.manifest["requirements"], [])

    def test_missing_and_unsafe_shards_rejected(self):
        index = self.source / "model.safetensors.index.json"
        for shard in ("missing.safetensors", "../outside.safetensors", "/absolute", "..\\outside"):
            with self.subTest(shard=shard):
                # Arrange
                index.write_text(json.dumps({"weight_map": {"weight": shard}}))

                # Act
                with self.assertRaises(ValueError) as raised:
                    inspect_model(self.source)

                # Assert
                self.assertIn("shard", str(raised.exception).lower())

    def test_nested_index_shards_are_copied(self):
        # Arrange
        (self.source / "shards").mkdir()
        (self.source / "shards/part.safetensors").write_bytes(b"nested")
        (self.source / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {"weight": "shards/part.safetensors"},
                }
            )
        )

        # Act
        result = self.build()

        # Assert
        self.assertEqual(
            (result.model_directory / "1/model/shards/part.safetensors").read_bytes(), b"nested"
        )

    def test_conflicting_task_rejected(self):
        # Arrange
        task = "text-classification"

        # Act
        with self.assertRaises(ValueError) as raised:
            self.build(task=task)

        # Assert
        self.assertIn("conflicts", str(raised.exception))

    def test_multimodal_checkpoint_rejected(self):
        # Arrange
        self.config["vision_config"] = {}
        self.write_config()

        # Act
        with self.assertRaises(ValueError) as raised:
            self.build()

        # Assert
        self.assertIn("Multimodal", str(raised.exception))

    def test_ambiguous_checkpoint_has_no_inferred_task(self):
        # Arrange
        self.config["architectures"] = ["BertModel"]
        self.write_config()

        # Act
        info = inspect_model(self.source)

        # Assert
        self.assertIsNone(info.inferred_task)

    def test_ambiguous_task_requires_explicit_selection(self):
        # Arrange
        self.config["architectures"] = ["BertModel"]
        self.write_config()

        # Act
        with self.assertRaises(ValueError) as raised:
            self.build()

        # Assert
        self.assertIn("Specify", str(raised.exception))

    def test_explicit_feature_extraction_task_is_accepted(self):
        # Arrange
        self.config["architectures"] = ["BertModel"]
        self.write_config()

        # Act
        result = self.build(task="feature-extraction")

        # Assert
        self.assertEqual(result.manifest["task"], "feature-extraction")

    def test_text2text_and_classification_task_inference(self):
        cases = [
            (
                {"architectures": ["T5ForConditionalGeneration"], "is_encoder_decoder": True},
                "text2text-generation",
            ),
            ({"architectures": ["BertForSequenceClassification"]}, "text-classification"),
        ]
        for config, expected in cases:
            with self.subTest(task=expected):
                # Arrange
                self.config = config
                self.write_config()

                # Act
                info = inspect_model(self.source)

                # Assert
                self.assertEqual(info.inferred_task, expected)

    def test_existing_output_preserved(self):
        # Arrange
        repository = self.root / "repository"
        repository.mkdir()
        marker = repository / "important"
        marker.write_text("keep me")

        # Act
        with self.assertRaises(ValueError) as raised:
            self.build()

        # Assert
        self.assertIn("already exists", str(raised.exception))
        self.assertEqual(marker.read_text(), "keep me")

    def test_nested_output_rejected(self):
        # Arrange
        output = self.source / "export"

        # Act
        with self.assertRaises(ValueError) as raised:
            generate_repository(self.source, output, name="demo", runtime="vllm", image="test:1")

        # Assert
        self.assertIn("outside", str(raised.exception))
        self.assertFalse(output.exists())

    def test_failed_copy_leaves_no_partial_repository(self):
        # Arrange
        with patch("triton_hf_builder.builder.shutil.copy2", side_effect=OSError("disk full")):
            # Act
            with self.assertRaisesRegex(OSError, "disk full"):
                self.build()

        # Assert
        self.assertFalse((self.root / "repository").exists())
        self.assertEqual(list(self.root.glob(".triton-hf-*")), [])

    @unittest.skipIf(sys.platform == "win32", "SIGTERM has different semantics on Windows")
    def test_cli_cancellation_cleans_up_partial_weight_copy(self):
        # Arrange
        script = """
import os, signal, sys
from unittest.mock import patch
from triton_hf_builder.cli import main
def cancel(*args, **kwargs):
    os.kill(os.getpid(), signal.SIGTERM)
with patch("triton_hf_builder.builder.shutil.copy2", side_effect=cancel):
    code = main(["build", sys.argv[1], "--output", sys.argv[2], "--name", "demo",
                 "--runtime", "vllm", "--image", "test:1"])
assert code == 130
"""

        # Act
        completed = subprocess.run(
            [sys.executable, "-c", script, str(self.source), str(self.root / "output")],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Assert
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("cancelled", completed.stderr)
        self.assertFalse((self.root / "output").exists())
        self.assertEqual(list(self.root.glob(".triton-hf-*")), [])

    def test_invalid_build_options(self):
        cases = [
            ({"device": "cpu"}, "GPU"),
            ({"tensor_parallel_size": 0}, "positive"),
            ({"image": "test:latest"}, "explicit"),
            ({"requirements": ["vllm==0.1"]}, "must come"),
            ({"requirements": ["bad name!"]}, "Invalid requirement"),
            ({"requirements": ["pkg @ https://example.com/pkg.whl"]}, "not URL"),
        ]
        for overrides, expected in cases:
            with self.subTest(overrides=overrides):
                # Arrange
                options = dict(name="demo", runtime="vllm", image="test:1") | overrides

                # Act
                with self.assertRaises(ValueError) as raised:
                    generate_repository(self.source, self.root / "repository", **options)

                # Assert
                self.assertIn(expected, str(raised.exception))

    def test_cli_json_errors_and_success(self):
        for command, expected in [
            (["inspect", str(self.source)], 0),
            (["inspect", "/missing/model"], 1),
        ]:
            # Arrange
            arguments = [sys.executable, "-m", "triton_hf_builder", *command]

            # Act
            result = subprocess.run(
                arguments,
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Assert
            self.assertEqual(result.returncode, expected, result.stderr)
            payload = json.loads(result.stdout if expected == 0 else result.stderr)
            self.assertIn("model_type" if expected == 0 else "error", payload)


class FakeArray:
    def __init__(self, values, shape=None):
        self.values = values
        self.shape = shape or (len(values),)

    def __getitem__(self, index):
        return self.values[index]


class AdapterTests(ModelFixture):
    def _load(self, result, framework_name, framework):
        # Only individual entries are patched: never restore the entire sys.modules.
        numpy = types.SimpleNamespace(array=lambda values, dtype: FakeArray(values))
        utils = types.SimpleNamespace(
            get_input_tensor_by_name=lambda request, name: request,
            Tensor=lambda name, value: (name, value),
            InferenceResponse=lambda **kwargs: kwargs,
            TritonError=lambda error: error,
        )
        for name, module in [
            ("numpy", numpy),
            ("triton_python_backend_utils", utils),
            (framework_name, framework),
        ]:
            # patch.dict on the whole module registry can unload lazy imports; restore
            # only the specific stub entries used by this test.
            previous = sys.modules.get(name)
            sys.modules[name] = module
            self.addCleanup(self._restore_module, name, previous)
        spec = importlib.util.spec_from_file_location(
            "generated_adapter", result.model_directory / "1/model.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.TritonPythonModel()

    @staticmethod
    def _restore_module(name, previous):
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous

    def test_transformers_adapter_device_prompt_and_request_errors(self):
        # Arrange
        result = self.build()
        calls = []

        def pipeline(**options):
            calls.append(options)

            def predict(prompt, **kwargs):
                calls.append((prompt, kwargs))
                return [{"generated_text": "completion"}]

            return predict

        model = self._load(result, "transformers", types.SimpleNamespace(pipeline=pipeline))
        # Environment flags are intentionally set by runtime initialization. Restore
        # these two entries individually so tests don't leak them into the process.
        old = {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        self.addCleanup(self._restore_environment, old)
        request = types.SimpleNamespace(as_numpy=lambda: FakeArray([b"hello"]))
        malformed = types.SimpleNamespace(as_numpy=lambda: FakeArray([b"a", b"b"]))

        # Act
        model.initialize(
            {
                "model_repository": str(result.model_directory),
                "model_version": "1",
                "model_instance_kind": "GPU",
                "model_instance_device_id": "2",
            }
        )
        responses = model.execute([request, malformed, None, request])
        model.finalize()

        # Assert
        self.assertEqual(calls[0]["device"], 2)
        self.assertFalse(calls[0]["trust_remote_code"])
        self.assertEqual(calls[1][0], "hello")
        self.assertFalse(calls[1][1]["return_full_text"])
        self.assertEqual(responses[0]["output_tensors"][0][1][0], b"completion")
        self.assertIn("error", responses[1])
        self.assertIn("error", responses[2])
        self.assertIn("output_tensors", responses[3])
        self.assertIsNone(model.pipeline)

    @staticmethod
    def _restore_environment(previous):
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_trtllm_adapter_uses_local_checkpoint_and_public_api(self):
        # Arrange
        result = self.build("tensorrt-llm", tensor_parallel_size=2)
        calls = []

        class Engine:
            def __init__(self, **kwargs):
                calls.append(kwargs)

            def generate(self, prompts, sampling_params):
                calls.append((prompts, sampling_params))
                return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text="answer")])]

            def shutdown(self):
                calls.append("shutdown")

        framework = types.SimpleNamespace(LLM=Engine, SamplingParams=lambda **kw: kw)
        model = self._load(result, "tensorrt_llm", framework)
        old = {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        self.addCleanup(self._restore_environment, old)
        request = types.SimpleNamespace(as_numpy=lambda: FakeArray([b"prompt"]))

        # Act
        model.initialize({"model_repository": str(result.model_directory), "model_version": "1"})
        response = model.execute([request])[0]
        model.finalize()

        # Assert
        self.assertEqual(calls[0]["model"], str(result.model_directory / "1/model"))
        self.assertEqual(calls[0]["backend"], "pytorch")
        self.assertEqual(calls[0]["tensor_parallel_size"], 2)
        self.assertEqual(response["output_tensors"][0][1][0], b"answer")
        self.assertEqual(calls[-1], "shutdown")


if __name__ == "__main__":
    unittest.main()
