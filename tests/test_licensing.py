"""License preservation with synthetic assets; no model downloads or Docker."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_builder import ModelFixture

from triton_hf_builder import generate_repository

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from serving_tests.checkpoints import download_checkpoint


class LicensingTests(ModelFixture):
    def test_checkpoint_attribution_documents_survive_export(self):
        # Arrange
        documents = {
            "LICENSE": "Synthetic checkpoint license\n",
            "LICENSE-MIT": "Synthetic alternative license\n",
            "NOTICE.md": "Synthetic third-party attribution\n",
            "COPYING.txt": "Synthetic copying terms\n",
            "README.md": "Synthetic model card\n",
            "model_card.rst": "Synthetic model documentation\n",
            "licence.markdown": "Synthetic lowercase license\n",
        }
        for name, content in documents.items():
            (self.source / name).write_text(content)

        # Act
        result = self.build()

        # Assert
        assets = result.model_directory / "1/model"
        for name, content in documents.items():
            with self.subTest(name=name):
                self.assertEqual((assets / name).read_text(), content)
        self.assertNotEqual((result.repository / "README.md").read_text(), documents["README.md"])

    def test_attribution_names_do_not_allow_executable_or_hidden_files(self):
        # Arrange
        excluded = ("LICENSE.py", "README.py", ".env", ".license", "unrelated.md")
        for name in excluded:
            (self.source / name).write_text("Do not export\n")

        # Act
        result = self.build()

        # Assert
        assets = result.model_directory / "1/model"
        self.assertTrue(all(not (assets / name).exists() for name in excluded))

    def test_all_adapters_ship_license_for_generated_support_code(self):
        # Arrange
        expected = (Path(__file__).resolve().parents[1] / "LICENSE").read_text()
        cases = (
            ("transformers", "demo/1/model.py"),
            ("tensorrt-llm", "demo/1/model.py"),
            ("vllm", "prepare_repository.py"),
        )
        for runtime, script in cases:
            with self.subTest(runtime=runtime):
                # Act
                result = generate_repository(
                    self.source,
                    self.root / runtime,
                    name="demo",
                    runtime=runtime,
                    image="test:1",
                )

                # Assert
                self.assertEqual(
                    (result.repository / "LICENSE.triton-hf-builder").read_text(), expected
                )
                self.assertTrue(
                    (result.repository / script)
                    .read_text()
                    .startswith("# SPDX-License-Identifier: Apache-2.0\n")
                )
                self.assertIn(
                    "assets retain their own licenses",
                    (result.repository / "README.md").read_text(),
                )

    def test_serving_download_preserves_attribution_at_the_resolved_revision(self):
        # Arrange
        names = (
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "README.md",
            "LICENSE",
            "NOTICE.md",
            "LICENSE.py",
            ".env",
        )
        info = SimpleNamespace(
            sha="a" * 40,
            siblings=[SimpleNamespace(rfilename=name) for name in names],
        )
        calls = []

        def snapshot(repo_id, **kwargs):
            calls.append((repo_id, kwargs))
            return str(self.source)

        hub = SimpleNamespace(
            HfApi=lambda: SimpleNamespace(model_info=lambda *args, **kwargs: info),
            snapshot_download=snapshot,
        )
        revisions = {}
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            # Act
            path, revision = download_checkpoint("example/synthetic", self.root, revisions)

        # Assert
        self.assertEqual(path, self.source)
        self.assertEqual(revision, info.sha)
        self.assertEqual(revisions, {"example/synthetic": info.sha})
        self.assertEqual(calls[0][1]["revision"], info.sha)
        patterns = calls[0][1]["allow_patterns"]
        self.assertTrue({"README.md", "LICENSE", "NOTICE.md"}.issubset(patterns))
        self.assertNotIn("LICENSE.py", patterns)
        self.assertNotIn(".env", patterns)
