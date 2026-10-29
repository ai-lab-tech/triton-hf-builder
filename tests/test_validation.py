from test_builder import ModelFixture

from triton_hf_builder import generate_repository
from triton_hf_builder.requirements import image_requirements


class ValidationTests(ModelFixture):
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
        ]
        for runtime, image in cases:
            with self.subTest(runtime=runtime, image=image):
                # Act
                extras = image_requirements(runtime, image)
                # Assert
                self.assertEqual(extras, ())

    def test_conditional_override_preserves_required_dependency(self):
        # Arrange
        requirements = ['torch==2.9.0; sys_platform == "win32"']
        # Act
        result = self.build(requirements=requirements)
        # Assert
        self.assertIn("torch", result.manifest["requirements"])
        self.assertIn(requirements[0], result.manifest["requirements"])

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
                    # Act
                    with self.assertRaises(ValueError) as error:
                        self.build(**{key: value})
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
