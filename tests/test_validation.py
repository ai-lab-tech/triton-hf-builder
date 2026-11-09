from unittest.mock import patch

from test_builder import ModelFixture

from triton_hf_builder import generate_repository
from triton_hf_builder.dependency_profiles import dependency_profile
from triton_hf_builder.requirements import image_requirements, resolve_requirements
from triton_hf_builder.runtimes import get_adapter


class ValidationTests(ModelFixture):
    def test_cpu_profile_preserves_pins_and_cpu_package_index(self):
        # Arrange
        (self.source / "tokenizer.model").write_bytes(b"tokenizer")
        # Act
        result = generate_repository(
            self.source,
            self.root / "repository",
            name="demo",
            runtime="transformers",
            device="cpu",
            image="nvcr.io/nvidia/tritonserver:26.06-py3",
        )
        # Assert
        self.assertIn("torch==2.9.0+cpu", result.manifest["requirements"])
        self.assertIn("transformers==4.57.1", result.manifest["requirements"])
        self.assertIn("sentencepiece==0.2.1", result.manifest["requirements"])
        self.assertNotIn("sentencepiece", result.manifest["requirements"])
        content = (result.repository / "requirements.txt").read_text()
        self.assertIn("--find-links https://download.pytorch.org/whl/cpu/torch/", content)

    def test_cpu_profile_records_installation_scope_without_claiming_serving_validation(self):
        # Arrange
        image = "nvcr.io/nvidia/tritonserver:26.06-py3"
        # Act
        result = generate_repository(
            self.source,
            self.root / "repository",
            name="demo",
            runtime="transformers",
            device="cpu",
            image=image,
        )
        # Assert
        self.assertEqual(result.manifest["dependency_profile"]["status"], "example_tested")
        self.assertEqual(result.manifest["validation"]["serving"], "not_run")
        self.assertEqual(result.manifest["validation"]["dependencies"], "unlocked")
        self.assertEqual(
            result.manifest["dependency_profile"]["installation"]["kind"], "isolated_python"
        )
        self.assertIn("isolated Python 3.12", (result.repository / "README.md").read_text())

    def test_checkpoint_transformers_version_does_not_override_profile(self):
        # Arrange
        self.config["transformers_version"] = "99.0.0"
        self.write_config()
        # Act
        result = generate_repository(
            self.source,
            self.root / "repository",
            name="demo",
            runtime="transformers",
            device="cpu",
            triton_version="26.06",
        )
        # Assert
        self.assertIn("transformers==4.57.1", result.manifest["requirements"])
        self.assertEqual(result.manifest["model"]["transformers_version"], "99.0.0")

    def test_partial_profile_preserves_required_adapter_dependencies(self):
        # Arrange
        profile = {"requirements": ["transformers==4.57.1"]}
        adapter = get_adapter("transformers")
        # Act
        dependencies = resolve_requirements(adapter, self.source, (), profile=profile)
        # Assert
        self.assertEqual(
            set(dependencies), {"numpy", "torch", "transformers==4.57.1", "safetensors"}
        )

    def test_explicitly_cataloged_digest_can_select_a_profile(self):
        # Arrange
        image = "nvcr.io/nvidia/tritonserver:26.06-py3@sha256:" + "a" * 64
        profile = {"runtime": "transformers", "image": image, "device": "cpu"}
        with patch(
            "triton_hf_builder.dependency_profiles._catalog", return_value={"profiles": [profile]}
        ):
            # Act
            selected = dependency_profile("transformers", image, "cpu")
        # Assert
        self.assertEqual(selected, profile)

    def test_conditional_profile_rule_cannot_remove_required_tokenizer_dependency(self):
        # Arrange
        (self.source / "tokenizer.model").write_bytes(b"tokenizer")
        conditional = 'sentencepiece==0.2.1; sys_platform == "win32"'
        profile = {"requirements": [conditional]}
        adapter = get_adapter("transformers")
        # Act
        dependencies = resolve_requirements(adapter, self.source, (), profile=profile)
        # Assert
        self.assertIn("sentencepiece", dependencies)
        self.assertIn(conditional, dependencies)

    def test_profile_pins_can_be_overridden_without_claiming_validation(self):
        # Arrange
        requirements = ["transformers==4.56.0"]
        # Act
        result = generate_repository(
            self.source,
            self.root / "repository",
            name="demo",
            runtime="transformers",
            device="cpu",
            image="nvcr.io/nvidia/tritonserver:26.06-py3",
            requirements=requirements,
        )
        # Assert
        self.assertIn("transformers==4.56.0", result.manifest["requirements"])
        self.assertNotIn("transformers==4.57.1", result.manifest["requirements"])
        self.assertTrue(any("overrides" in warning for warning in result.manifest["warnings"]))

    def test_profiles_are_not_inferred_for_unknown_images_devices_or_backends(self):
        # Arrange
        cases = [
            ("transformers", "nvcr.io/nvidia/tritonserver:26.06-py3", "gpu"),
            ("transformers", "nvcr.io/nvidia/tritonserver:99.01-py3", "cpu"),
            ("transformers", "custom/triton:26.06-py3", "cpu"),
            ("transformers", "nvcr.io/nvidia/tritonserver@sha256:" + "a" * 64, "cpu"),
            ("transformers", "nvcr.io/nvidia/tritonserver:26.06-py3@sha256:" + "a" * 64, "cpu"),
            ("vllm", "nvcr.io/nvidia/tritonserver:26.06-py3", "gpu"),
        ]
        for index, (runtime, image, device) in enumerate(cases):
            with self.subTest(runtime=runtime, image=image, device=device):
                # Arrange
                output = self.root / f"repository-{index}"
                # Act
                result = generate_repository(
                    self.source,
                    output,
                    name="demo",
                    runtime=runtime,
                    image=image,
                    device=device,
                    requirements=["torch==2.9.0"] if runtime == "transformers" else (),
                )
                # Assert
                self.assertIsNone(result.manifest["dependency_profile"])
                self.assertEqual(result.manifest["package_indexes"], [])
                self.assertTrue(
                    any("No bundled dependency profile" in w for w in result.manifest["warnings"])
                )

    def test_engine_profile_records_published_stack_without_installing_it(self):
        # Arrange
        image = "nvcr.io/nvidia/tritonserver:26.06-vllm-python-py3"
        # Act
        result = generate_repository(
            self.source, self.root / "repository", name="demo", runtime="vllm", image=image
        )
        # Assert
        profile = result.manifest["dependency_profile"]
        self.assertEqual(profile["status"], "manufacturer_documented")
        self.assertEqual(profile["release"], "26.06")
        self.assertTrue(profile["environment"]["engine_version"])
        self.assertTrue(profile["sources"][0]["commit"])
        self.assertEqual(result.manifest["requirements"], [])
        self.assertEqual(result.manifest["validation"]["serving"], "not_run")

    def test_trt_2606_requirements_include_verified_sdk_pin(self):
        # Arrange
        image = "nvcr.io/nvidia/tritonserver:26.06-trtllm-python-py3"
        # Act
        result = generate_repository(
            self.source, self.root / "repository", name="demo", runtime="tensorrt-llm", image=image
        )
        # Assert
        self.assertIn("openai==2.8.1", result.manifest["requirements"])
        self.assertIn("openai==2.8.1", (result.repository / "requirements.txt").read_text())

    def test_trt_sdk_pin_can_be_explicitly_overridden(self):
        # Arrange
        image = "nvcr.io/nvidia/tritonserver:26.06-trtllm-python-py3"
        # Act
        result = generate_repository(
            self.source,
            self.root / "repository",
            name="demo",
            runtime="tensorrt-llm",
            image=image,
            requirements=["openai==2.8.2"],
        )
        # Assert
        self.assertIn("openai==2.8.2", result.manifest["requirements"])
        self.assertNotIn("openai==2.8.1", result.manifest["requirements"])

    def test_image_correction_is_not_guessed_for_other_images_or_runtimes(self):
        # Arrange
        cases = [
            ("tensorrt-llm", "nvcr.io/nvidia/tritonserver:26.07-trtllm-python-py3"),
            ("tensorrt-llm", "custom/triton:26.06-trtllm-python-py3"),
            ("vllm", "nvcr.io/nvidia/tritonserver:26.06-trtllm-python-py3"),
            ("tensorrt-llm", "nvcr.io/nvidia/tritonserver@sha256:" + "a" * 64),
            (
                "tensorrt-llm",
                "nvcr.io/nvidia/tritonserver:26.06-trtllm-python-py3@sha256:" + "a" * 64,
            ),
        ]
        for runtime, image in cases:
            with self.subTest(runtime=runtime, image=image):
                # Arrange
                arguments = {"runtime": runtime, "image": image}
                # Act
                extras = image_requirements(**arguments)
                # Assert
                self.assertEqual(extras, ())

    def test_conditional_override_preserves_required_dependency(self):
        # Arrange
        requirements = ['torch==2.9.0; sys_platform == "win32"']
        # Act
        result = self.build(requirements=requirements)
        # Assert
        self.assertIn("torch", result.manifest["requirements"])
        self.assertIn(
            'torch!=2.9.0+cpu,==2.9.0; sys_platform == "win32"', result.manifest["requirements"]
        )

    def test_platform_alternatives_are_preserved(self):
        # Arrange
        requirements = ['extra==1; sys_platform == "win32"', 'extra==2; sys_platform == "linux"']
        # Act
        result = self.build(requirements=requirements)
        # Assert
        self.assertTrue(set(requirements).issubset(result.manifest["requirements"]))

    def test_integer_options_reject_boolean_and_fractional_values(self):
        # Arrange
        for key in ("tensor_parallel_size", "max_new_tokens", "max_model_len"):
            for value in (True, 1.5):
                with self.subTest(option=key, value=value):
                    # Arrange
                    options = {key: value}
                    # Act
                    with self.assertRaises(ValueError) as error:
                        self.build(**options)
                    # Assert
                    self.assertIn("positive integer", str(error.exception))

    def test_requirements_string_is_not_treated_as_characters(self):
        # Arrange
        requirements = "torch==2.9.0"
        # Act
        with self.assertRaises(ValueError) as error:
            self.build(requirements=requirements)
        # Assert
        self.assertIn("sequence", str(error.exception))


"""Validate dependency profiles, requirement overrides and generation options."""
