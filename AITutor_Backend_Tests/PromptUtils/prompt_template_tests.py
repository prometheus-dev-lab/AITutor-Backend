import unittest
from unittest.mock import mock_open, patch
from AITutor_Backend.src.PromptUtils.prompt_template import PromptTemplate
import os
import shutil


class TestPromptTemplate(unittest.TestCase):

    def setUp(self):
        self.prompt_data = "Hello, {name}. Welcome to {place}!"
        self.variables = {"name": "{name}", "place": "{place}"}

    def test_initialization(self):
        """Test if the PromptTemplate initializes correctly with variable checks"""
        self.assertIsInstance(
            PromptTemplate(self.prompt_data, self.variables), PromptTemplate
        )

    def test_replace(self):
        # Testing replacements
        result = PromptTemplate(self.prompt_data, self.variables).replace(
            name="Alice", place="Wonderland"
        )
        self.assertEqual(result, "Hello, Alice. Welcome to Wonderland!")

    def test_prompt_replacement_missing_variable(self):
        """Test errors on missing replacement variable"""
        template = PromptTemplate(self.prompt_data, self.variables)

        with self.assertRaises(AssertionError):
            template.replace(name="Alice")

    def test_from_file(self):
        """Test creating a PromptTemplate from a file"""
        prompt_content = self.prompt_data
        prompt_file_path = "./tmp/"
        prompt_file_name = "dummy_prompt.txt"

        # Make the tmp folder
        if not os.path.exists(prompt_file_path):
            os.makedirs(prompt_file_path)

        # Write data to the prompt file
        with open(os.path.join(prompt_file_path, prompt_file_name), "w") as f:
            f.write(prompt_content)

        # Load from the path and ensure data is correct:
        template = PromptTemplate.from_file(
            os.path.join(prompt_file_path, prompt_file_name), self.variables
        )
        self.assertEqual(template._prompt_data, prompt_content)

        # Clean up tmp data
        shutil.rmtree(prompt_file_path)


if __name__ == "__main__":
    unittest.main()
