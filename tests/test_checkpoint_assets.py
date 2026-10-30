"""Checkpoint metadata must not bypass the export's file filters."""

import json

from test_builder import ModelFixture

from triton_hf_builder import generate_repository, inspect_model


class CheckpointAssetTests(ModelFixture):
    def test_index_cannot_export_non_weight_files(self):
        for filename in ("custom_model.py", ".env", "credentials.json", "notes.txt", "other.bin"):
            with self.subTest(filename=filename):
                # Arrange
                (self.source / filename).write_text("not checkpoint weights")
                (self.source / "model.safetensors.index.json").write_text(
                    json.dumps({"weight_map": {"weight": filename}})
                )
                output = self.root / f"export-{filename}"

                # Act
                with self.assertRaises(ValueError) as error:
                    generate_repository(
                        self.source, output, name="demo", runtime="vllm", image="test:1"
                    )

                # Assert
                self.assertIn("shard", str(error.exception).lower())
                self.assertFalse(output.exists())

    def test_hidden_root_weights_are_not_exported(self):
        # Arrange
        (self.source / ".private.safetensors").write_bytes(b"private cached data")

        # Act
        result = self.build()

        # Assert
        assets = result.model_directory / "1/model"
        self.assertFalse((assets / ".private.safetensors").exists())
        self.assertTrue((assets / "model.safetensors").is_file())

    def test_index_cannot_export_weights_from_hidden_directories(self):
        # Arrange
        private = self.source / ".cache"
        private.mkdir()
        (private / "private.safetensors").write_bytes(b"private cached data")
        (self.source / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": ".cache/private.safetensors"}})
        )

        # Act
        with self.assertRaises(ValueError) as error:
            self.build()

        # Assert
        self.assertIn("Unsafe shard", str(error.exception))
        self.assertFalse((self.root / "repository").exists())

    def test_windows_drive_shards_are_rejected_on_every_platform(self):
        for shard in ("C:/weights.safetensors", "C:weights.safetensors"):
            with self.subTest(shard=shard):
                # Arrange
                (self.source / "model.safetensors.index.json").write_text(
                    json.dumps({"weight_map": {"weight": shard}})
                )

                # Act
                with self.assertRaises(ValueError) as error:
                    inspect_model(self.source)

                # Assert
                self.assertIn("Unsafe shard", str(error.exception))

    def test_pytorch_index_copies_nested_bin_shards(self):
        # Arrange
        (self.source / "model.safetensors").unlink()
        (self.source / "shards").mkdir()
        (self.source / "shards/part.bin").write_bytes(b"fixture-weight-bytes")
        (self.source / "pytorch_model.bin.index.json").write_text(
            json.dumps({"weight_map": {"weight": "shards/part.bin"}})
        )

        # Act
        result = self.build()

        # Assert
        self.assertEqual(
            (result.model_directory / "1/model/shards/part.bin").read_bytes(),
            b"fixture-weight-bytes",
        )

    def test_only_jinja_templates_are_copied(self):
        # Arrange
        templates = self.source / "chat_templates"
        templates.mkdir()
        for filename in ("chat.jinja", "chat.jinja2", "custom.jinja.py", ".hidden.jinja"):
            (templates / filename).write_text("fixture")

        # Act
        result = self.build()

        # Assert
        exported = result.model_directory / "1/model/chat_templates"
        self.assertEqual({p.name for p in exported.iterdir()}, {"chat.jinja", "chat.jinja2"})
