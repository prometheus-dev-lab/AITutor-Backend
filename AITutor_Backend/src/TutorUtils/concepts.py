import os
import re
import threading
from typing import List, Tuple, Optional, Union
import yaml

from AITutor_Backend.src.BackendUtils.json_serialize import JSONSerializable
from AITutor_Backend.src.BackendUtils.replicate_api import ReplicateAPI
from AITutor_Backend.src.BackendUtils.sql_serialize import SQLSerializable

from AITutor_Backend.src.BackendUtils.llm_client import LLM

from AITutor_Backend.src.PromptUtils.prompt_template import PromptTemplate
from AITutor_Backend.src.PromptUtils.prompt_utils import (
    Message,
    Conversation,
    AI_TUTOR_MSG,
)

from AITutor_Backend.src.DataUtils.file_utils import save_training_data
from AITutor_Backend.src.DataUtils.nlp_utils import edit_distance

USE_OPENAI = True
MIN_EDIT_DISTANCE_SIMILARITY = 4
DEBUG = bool(os.environ.get("DEBUG", 0))


class ConceptDatabase(SQLSerializable, JSONSerializable):
    @staticmethod
    def build_dict(lines, current_level=0, index=0):
        """
        Recursively build a dictionary structure from a list of lines.
        """

        def parse_line(l):
            """
            Parse a line to determine its concept and indentation level.
            Indentation level is determined by the number of leading spaces.
            """
            indentation = len(l) - len(l.lstrip("\t"))
            concept = l.strip()
            return concept, indentation

        if index >= len(lines):
            return [], index

        tree = []
        while index < len(lines):
            concept, level = parse_line(lines[index])
            if level > current_level:
                # If the next level is deeper, recursively build its structure
                children, index = ConceptDatabase.build_dict(lines, level, index)
                if tree and "refs" in tree[-1]:
                    tree[-1]["refs"].extend(children)
                elif tree:
                    tree[-1]["refs"] = children
            elif level < current_level:
                # If the next level is higher, return to the previous level
                return tree, index
            else:
                # Same level, add a new node and continue
                tree.append({"concept": concept, "refs": []})
                index += 1

        return tree, index

    class ConceptPrompts:
        CURR_ENV_MAIN_CONCEPT_DELIMITER = (
            "$CURR_ENV.MAIN_CONCEPT$"  # TODO: Add tutor plan string to the llm request
        )
        CURR_ENV_CONCEPT_LIST_DELIMITER = "$CURR_ENV.CONCEPT_GRAPH$"
        CURR_ENV_CONTEXT_CONCEPT_DELIMITER = "$CURR_ENV.CONTEXT_CONCEPT$"
        CONCEPT_NAME_DELIMITER = "$TARGET_CONCEPT_NAME$"
        TUTOR_PLANNER_DELIMITER = "$CURR_ENV.TUTOR_PLANNER$"
        WIKI_DELIMITER = "$CONCEPT_WIKI_DATA$"
        CURR_ERROR_DELIMITER = "$CURR_ENV.ERROR$"

        def __init__(self, tutor_plan):
            self.__tutor_plan = tutor_plan
            # Initialize Concept Graph Generation prompt
            self._concept_graph_prompt_template = PromptTemplate.from_config(
                "@conceptGraph",
                {
                    "tutor_plan": ConceptDatabase.ConceptPrompts.TUTOR_PLANNER_DELIMITER,  # TODO: Optimize by embedding into the prompt
                    "main_concept": ConceptDatabase.ConceptPrompts.CURR_ENV_MAIN_CONCEPT_DELIMITER,
                },
            )
            """vars: tutor_plan, main_concept"""

            # Initialize Concept Generation Prompt
            self._concept_prompt_template = PromptTemplate.from_config(
                "@conceptPrompt",
                {
                    # "tutor_plan": self.TUTOR_PLANNER_DELIMITER,
                    "main_concept": self.CURR_ENV_MAIN_CONCEPT_DELIMITER,
                    "concept_list": self.CURR_ENV_CONTEXT_CONCEPT_DELIMITER,
                    "concept_name": self.CONCEPT_NAME_DELIMITER,
                    "curr_error": self.CURR_ERROR_DELIMITER,
                },
            )
            """vars: tutor_plan, main_concept, concept_list, concept_name, curr_error"""

        def request_concept_data_from_llm(
            self, env_main_concept, context_concept, wiki_data, concept_name, error_msg
        ) -> str:
            """Requests the Concept information from an LLM.

            Args:
                env_main_concept (str): main concept of CG
                env_concept_list (str): list of relevant and identified concepts
                concept_name (str): concept we are generating for

            Returns:
                str: LLM Output containing Concept Data
            """
            # TODO: Implement wiki data usage
            prompt = self._concept_prompt_template.replace(
                # tutor_plan=self.__tutor_plan,
                main_concept=env_main_concept,
                concept_list=context_concept,
                concept_name=concept_name,
                curr_error=error_msg,
            )

            # Create conversation with AI Tutor Sys Prompt.
            messages = Conversation.from_message_list(
                [AI_TUTOR_MSG, Message("user", prompt)]
            )

            llm_output = LLM("@conceptPrompt").chat_completion(messages)

            return prompt, llm_output

        def request_concept_graph_from_llm(self, env_main_concept, temp=0.5):
            """Requests the Concept information from an LLM.
            Args:
                env_main_concept (str): _description_

            Returns:
                str: LLM Output containing Concept Graph Data
            """
            prompt = self._concept_graph_prompt_template.replace(
                tutor_plan=self.__tutor_plan, main_concept=env_main_concept
            )

            messages = Conversation.from_message_list(
                [AI_TUTOR_MSG, Message("user", prompt)]
            )

            llm_output = LLM("@conceptGraph").chat_completion(messages)

            return prompt, llm_output

    __CONCEPT_REGEX = re.compile(
        r"\`\`\`yaml([^\`]*)\`\`\`"
    )  # Matches any ```yaml ...```
    __GRAPH_REGEX = re.compile(
        r"\`\`\`plaintext([^\`]*)\`\`\`"
    )  # Matches any ```plaintext ... ```

    def __init__(
        self, main_concept: str, tutor_plan: str = "", max_threads=120
    ):  # TODO: Fix potential Resource lock issue
        self.concept_llm_api = self.ConceptPrompts(
            tutor_plan=tutor_plan,
        )  # TODO: FIX
        self.main_concept = main_concept
        self.concepts = {}

    def generate_concept_graph(
        self,
    ):
        """
        Generates a Concept Graph from a LLM for the main concept
        """
        while True:
            prompt, llm_output = self.concept_llm_api.request_concept_graph_from_llm(
                self.main_concept,
            )
            try:
                # with open("translation.txt", "a") as f:
                #     f.write("TRANSLATION\n")
                regex_match = ConceptDatabase.__GRAPH_REGEX.findall(llm_output)
                assert (
                    regex_match
                ), f"Error parsing LLM Output for Concept Graph. Ensure you properly used the Yaml Creation rules."

                # Extract the Yaml Data from the LLM Ouput
                data = regex_match[0].replace("    ", "\t").strip()
                graph_data = ConceptDatabase.build_dict(data.split("\n"))
                assert (
                    graph_data
                ), "Could not create concept from LLM Output, ensure you have properly entered the information and did not include any additional information outside of what's required."

                # Save the llm_output as training tata to file "training_data/concept/generation/":
                output_dir = "training_data/concept/graph_generation/"
                save_training_data(output_dir, prompt, llm_output)
                break
            except Exception as e:
                error_msg = str(e)  # TODO: LOG
                # with open("translation_errors.txt", "a") as f:
                #     f.write("TRANSLATION_ERROR\n")

        # Recursively add concepts to the graph
        def add_concepts_recursively(c, parent: Concept, cd: ConceptDatabase):
            concept = Concept(name=c["concept"], parent=parent)
            cd.concepts[concept.name] = concept
            if parent is not None:
                parent.refs.append(concept)
            for c_ref in c["refs"]:
                add_concepts_recursively(c_ref, concept, cd)

        for concept in graph_data[0]:
            add_concepts_recursively(concept, None, self)

        if DEBUG:
            print("Concept Graph:\n", self.get_concept_graph_str())

    def get_concept_list_str(
        self,
    ):
        return (
            "\n".join(
                [f'\t- "{concept.name}"' for concept in self.concepts.values()],
            )
            if self.concepts
            else "The Concept List is Empty."
        )

    def get_concept_graph_str(
        self,
    ):
        # Contained parents of the CG
        origin_nodes = [c for c in self.concepts.values() if c.parent == None]
        c_graph = ""

        # Recursively generates an indented representation of the CG
        def rec_gen_c_graph(node: Concept, tab=1):
            return (
                f"{node.name}\n"
                + "\n".join(
                    ["\t" * tab + rec_gen_c_graph(c, tab + 1) for c in node.refs]
                )
                if node != None
                else ""
            )

        # Append each local graph together into a global graph
        for origin_node in origin_nodes:
            c_graph += rec_gen_c_graph(origin_node)

        return c_graph

    def get_concept(self, concept_name: str):
        """Retrieves a concept from the concept graph"""
        if not concept_name:
            return None

        # Check if theres a direct match
        if concept_name in self.concepts:
            return self.concepts[concept_name]

        # Check if we can map using similarity. We use edit distance of 4 as a max such that concepts with almost identical spelling will be identical
        concept = [
            (edit_distance(concept_name.lower(), concept.name.lower()), concept)
            for concept in self.concepts.values()
            if edit_distance(concept_name.lower(), concept.name.lower())
            < MIN_EDIT_DISTANCE_SIMILARITY
        ]

        if concept:
            concept.sort(key=lambda x: x[0])  # Get the best match
            return concept[0][1]

        return None

    def generate_concepts(
        self,
    ):
        """Generates the concept data for each concept.
        This operation is threaded such that for each concept, it executes in parallel to speed up generation process.
        """
        if DEBUG:
            print("Generating Concepts...")

        threads = []
        generation_lock = threading.Lock()
        for concept_ref in self.concepts.values():
            thread = threading.Thread(
                target=self.generate_concept, args=(concept_ref, generation_lock)
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

    def generate_concept(self, concept: "Concept", generation_lock: threading.Lock):
        """
        Generates a Concept from a LLM
        """
        if DEBUG:
            print("Generating Concept:", concept.name)

        def escape_backslashes_in_quotes(text):
            """Escapes backslashes in quotes to prevent generation failures.

            Issue:
            Yaml data generation which arose from the LLM having to escape all quotations in the YAML data.

            Solution:
            We escape the quotes for the LLM, this simple solution fixes this problem"""

            def replace_backslashes(match):
                return match.group(0).replace("\\", "\\\\")

            quoted_strings_regex = r'"([^"\\]*(?:\\.[^"\\]*)*)"'
            return re.sub(quoted_strings_regex, replace_backslashes, text)

        error_msg = "There is no current error detected in the parsing system."
        context_concept = (
            concept.parent.name if concept.parent is not None else self.main_concept
        )

        # Validation Loop: Continue iteration until validated concept data is obtained, then exit
        while True:
            prompt, llm_output = self.concept_llm_api.request_concept_data_from_llm(
                self.main_concept,
                context_concept,
                "Not available.",
                concept.name,
                error_msg,
            )

            try:
                with open("translation.txt", "a") as f:
                    f.write("TRANSLATION\n")
                regex_match = ConceptDatabase.__CONCEPT_REGEX.findall(llm_output)
                assert (
                    regex_match
                ), f"Error parsing LLM Output for Concept: {concept.name}. Ensure you properly used the Yaml Creation rules."

                # Extract the Yaml Data from the LLM Ouput
                parsed_data = escape_backslashes_in_quotes(regex_match[0])
                parsed_data = yaml.safe_load(parsed_data)

                # Type check:
                assert isinstance(parsed_data, dict), "Root should be a dictionary"

                # Key check:
                assert (
                    "Concept" in parsed_data
                ), "Concept field is missing in the YAML data"

                assert all(
                    key in parsed_data["Concept"]
                    for key in [
                        "name",
                        "definition",
                    ]
                ), "Some required keys are missing in Concept"

                # Extract info from LLM Output:
                c_def = parsed_data["Concept"]["definition"]
                c_latex = parsed_data["Concept"].get("latex_code", None)
                if c_latex and (c_latex.lower() == "none" or c_latex.lower() == "null"):
                    c_latex = ""
                concept.definition = c_def
                concept.latex = c_latex

                assert (
                    concept.definition
                ), "Could not create concept from LLM Output, ensure you have properly entered the information and did not include any additional information outside of what's required."
                # Save the llm_output as training tata to file "training_data/concept/generation/":

                output_dir = "training_data/concept/generation/"
                save_training_data(output_dir, prompt, llm_output)
                break

            except Exception as e:
                print(e)
                error_msg = str(e)
                with open("translation_errors.txt", "a") as f:
                    f.write("TRANSLATION_ERROR\n")

    def format_json(self):
        return {
            "main_concept": self.main_concept,
            "tutor_plan": self.tutor_plan,
            "concepts": {
                name: concept.format_json() for name, concept in self.concepts.items()
            },
        }

    def to_sql(self):
        return self.format_json()

    @staticmethod
    def from_sql(data: dict) -> "ConceptDatabase":
        """creates a ConceptDatabase from sql data.

        Args:
            data (dict): data from sql

        Returns:
            ConceptDatabase
        """
        cd = ConceptDatabase(data["main_concept"], data["tutor_plan"])
        for concept_data in data["concepts"].values():
            concept = Concept.from_dict(concept_data)
            cd.concepts[concept.name] = concept

        for concept in cd.concepts.values():
            concept.parent = cd.get_concept(concept.parent)
            if concept.refs:
                concept.refs = [cd.get_concept(ref) for ref in concept.refs]
                concept.refs = [ref for ref in concept.refs if ref is not None]
        return cd

    @staticmethod # TODO: update concept database model
    def from_sql(main_concept, tutor_plan, concepts):
        """creates a ConceptDatabase from sql data.

        Args:
            main_concept (str): _description_
            tutor_plan (str): plan from tutor
            concepts (List[str, str, str, str, str]): List[(concept_name, parent.name, concept_def, concept_latex, refs_str)]

        Returns:
            _type_: ConceptDatabase
        """
        cd = ConceptDatabase(
            main_concept,
            tutor_plan,
        )

        cd.Concepts = [
            Concept.from_sql(
                cpt[0],
                cpt[1],
                cpt[2],
                cpt[3],
                (cpt[4].split("[SEP]") if cpt[4] is not None else []),
            )
            for cpt in concepts
        ]
        for (
            concept
        ) in cd.Concepts:  # Update/Populate references from the temporary references
            concept.parent = cd.get_concept(concept.parent)
            if concept.refs:
                concept.refs = [cd.get_concept(ref) for ref in concept.refs]
                concept.refs = [
                    ref for ref in concept.refs if ref is not None
                ]  # Filter None-references

        return cd

    def to_sql(
        self,
    ):
        """
        Returns:
            main_concept (str):
            concepts (List[str, str, str]): List[(concept_name, concept_def, concept_latex)]
        """
        return (self.main_concept, [c.to_sql() for c in self.Concepts])


class Concept(JSONSerializable):
    def __init__(
        self,
        name: str,
        parent: Optional[str] = None,
    ):
        self.name = name
        self.parent: Union[Concept, str, None] = parent
        self.definition = ""
        self.latex = ""
        self.refs = []

    def __repr__(self) -> str:
        return f"Concept(name: {self.name}, definition: {(self.definition[:50] + '...') if len(self.definition) > 50 else self.definition})"

    def format_json(self):
        return {
            "name": self.name,
            "parent": self.parent.name if self.parent is not None else None,
            "definition": self.definition,
            "latex": self.latex,
            "refs": [r.name for r in self.refs],
        }

    @staticmethod
    def from_dict(data: dict):
        n_concept = Concept(
            name=data["name"],
            parent=data["parent"],
        )
        n_concept.definition = data["definition"]
        n_concept.latex = data["latex"]
        n_concept.refs = data["refs"]
