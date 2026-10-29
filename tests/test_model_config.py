from dataclasses import replace
from unittest.mock import patch

from google.protobuf import text_format
from test_builder import ModelFixture
from tritonclient.grpc import model_config_pb2

from triton_hf_builder import generate_repository
from triton_hf_builder.model_config import parse_model_config
from triton_hf_builder.runtimes import ADAPTERS


class ModelConfigTests(ModelFixture):
    def test_all_adapters_generate_official_model_config_messages(self):
        # Arrange
        cases = [
            ("transformers", "cpu", "python", "KIND_CPU"),
            ("transformers", "gpu", "python", "KIND_GPU"),
            ("vllm", "gpu", "vllm", "KIND_MODEL"),
            ("tensorrt-llm", "gpu", "python", "KIND_CPU"),
        ]
        for runtime, device, backend, kind in cases:
            with self.subTest(runtime=runtime, device=device):
                # Act
                result = generate_repository(
                    self.source,
                    self.root / f"{runtime}-{device}",
                    name="demo",
                    runtime=runtime,
                    image="test:1",
                    device=device,
                )
                config = text_format.Parse(
                    (result.model_directory / "config.pbtxt").read_text(),
                    model_config_pb2.ModelConfig(),
                )
                # Assert
                self.assertEqual(config.name, "demo")
                self.assertEqual(config.backend, backend)
                self.assertEqual(config.max_batch_size, 0)
                self.assertEqual(len(config.instance_group), 1)
                self.assertEqual(config.instance_group[0].count, 1)
                self.assertEqual(
                    config.instance_group[0].kind,
                    model_config_pb2.ModelInstanceGroup.Kind.Value(kind),
                )
                if backend == "python":
                    self.assertEqual(len(config.input), 1)
                    self.assertEqual(config.input[0].name, "text_input")
                    self.assertEqual(config.input[0].data_type, model_config_pb2.TYPE_STRING)
                    self.assertEqual(list(config.input[0].dims), [1])
                    self.assertEqual(len(config.output), 1)
                    self.assertEqual(config.output[0].name, "text_output")
                    self.assertEqual(config.output[0].data_type, model_config_pb2.TYPE_STRING)
                    self.assertEqual(list(config.output[0].dims), [1])
                else:
                    self.assertEqual(len(config.input), 0)
                    self.assertEqual(len(config.output), 0)

    def test_invalid_protobuf_text_is_rejected(self):
        # Arrange
        cases = [
            ('name: "demo" unknown_field: true', "unknown_field"),
            ('input { name: "x" data_type: TYPE_NOT_REAL }', "TYPE_NOT_REAL"),
            ('input { dims: "wrong" }', "wrong"),
            ('max_batch_size: "wrong"', "wrong"),
            ('name: "unterminated', "Invalid config.pbtxt"),
        ]
        for content, detail in cases:
            with self.subTest(content=content):
                # Act
                with self.assertRaises(ValueError) as error:
                    parse_model_config(content)
                # Assert
                self.assertIn(detail, str(error.exception))

    def test_invalid_instance_kind_leaves_no_published_repository(self):
        # Arrange
        invalid = replace(ADAPTERS["transformers"], instance_kind="KIND_TYPO")
        with patch.dict(ADAPTERS, {"transformers": invalid}):
            # Act
            with self.assertRaises(ValueError) as error:
                self.build()
        # Assert
        self.assertIn("KIND_TYPO", str(error.exception))
        self.assertFalse((self.root / "repository").exists())
        self.assertEqual(list(self.root.glob(".triton-hf-*")), [])
