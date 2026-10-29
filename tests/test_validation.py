from test_builder import ModelFixture


class ValidationTests(ModelFixture):
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
