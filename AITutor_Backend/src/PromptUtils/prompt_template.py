import os
from typing import Any, Dict, Tuple

from AITutor_Backend.src.PromptUtils.prompt_configs import get_prompt_config


class PromptTemplate(object):
    def __init__(
        self,
        prompt_data: str,  # Templated prompt data
        variables: Dict[str, str],  # (Var Name -> Replacement Tag)
    ) -> None:
        """
        Initializes with Templated prompt Data, variables dict. Expects variables: {var_name: replacement_tag}
        """
        self._prompt_data = prompt_data
        self._variables = variables

        for var, repl_tag in self._variables.items():
            prompt_test = self._prompt_data
            prompt_test = prompt_test.replace(repl_tag, repl_tag + repl_tag)
            assert (
                prompt_test != self._prompt_data
            ), f"PromptTemplate Error: input Variable {var} does not replace anything in the prompt."

    @classmethod
    def from_file(
        cls,
        prompt_file_path: str,
        variables: Dict[str, str],  # (Var Name -> Replacement Tag)
    ) -> "PromptTemplate":
        """
        Creates a prompt template from a file given a specific dictionary of variables
        """
        assert os.path.exists(
            prompt_file_path
        ), f"[PromptTemplate] Creation Assertion Error: path does not exist, {prompt_file_path}"

        with open(prompt_file_path, "r", encoding="utf-8") as f:
            prompt_data = f.read()

        return cls(prompt_data, variables)

    @classmethod
    def from_config(
        cls,
        config_name: str,
        variables: Dict[str, str],  # (Var Name -> Replacement Tag)
    ) -> "PromptTemplate":
        """
        Creates a prompt template from the parsed config name. Ensure the name starts with @ and is recognized by the prompt_configs.py file.
        """
        prompt_file_path = get_prompt_config(config_name)
        assert os.path.exists(
            prompt_file_path
        ), f"PromptTemplate Creation Assertion Error: path does not exist, {prompt_file_path}"

        with open(prompt_file_path, "r", encoding="utf-8") as f:
            prompt_data = f.read()

        return cls(prompt_data, variables)

    @classmethod
    def from_prompt_str(
        cls,
        prompt_data: str,
        variables: Dict[str, str],
    ):
        """Clsmethod alias to the constructor."""
        return cls(prompt_data, variables)

    def replace(self, **kwargs):
        """"""
        # Create new prompt val for replacement
        prompt_val = self._prompt_data
        for var, repl_tag in self._variables.items():
            var_data = kwargs.get(var)
            assert (
                var_data is not None
            ), f"PromptTemplate Error: variable {var} not provided during replacement"
            prompt_val = prompt_val.replace(repl_tag, var_data)

        return prompt_val
