import os
import re
import threading
from typing import List, Tuple

import openai
import yaml

from AITutor_Backend.src.BackendUtils.json_serialize import JSONSerializable
from AITutor_Backend.src.BackendUtils.replicate_api import ReplicateAPI
from AITutor_Backend.src.BackendUtils.sql_serialize import SQLSerializable
from AITutor_Backend.src.DataUtils.file_utils import save_training_data
from AITutor_Backend.src.DataUtils.nlp_utils import edit_distance

USE_OPENAI = True
DEBUG = bool(os.environ.get("DEBUG", 0))

class ConceptDatabase(SQLSerializable):
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
            indentation = len(l) - len(l.lstrip('\t'))
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
                if tree and 'refs' in tree[-1]:
                    tree[-1]['refs'].extend(children)
                elif tree:
                    tree[-1]['refs'] = children
            elif level < current_level:
                # If the next level is higher, return to the previous level
                return tree, index
            else:
                # Same level, add a new node and continue
                tree.append({'concept': concept, 'refs': []})
                index += 1

        return tree, index
    
    class ConceptLLMAPI:
        CURR_ENV_MAIN_CONCEPT_DELIMITER = "$CURR_ENV.MAIN_CONCEPT$" # TODO: Add tutor plan string to the llm request
        CURR_ENV_CONCEPT_LIST_DELIMITER = "$CURR_ENV.CONCEPT_GRAPH$"
        CURR_ENV_CONTEXT_CONCEPT_DELIMITER = "$CURR_ENV.CONTEXT_CONCEPT$"
        CONCEPT_NAME_DELIMITER = "$TARGET_CONCEPT_NAME$"
        TUTOR_PLANNER_DELIMITER = "$CURR_ENV.TUTOR_PLANNER$"
        WIKI_DELIMITER = "$CONCEPT_WIKI_DATA$"
        CURR_ERROR_DELIMITER = "$CURR_ENV.ERROR$"


        def __init__(self, concept_graph_prompt_file, concept_prompt_file, tutor_plan):
            self.__tutor_plan = tutor_plan
            with open(concept_graph_prompt_file, "r", encoding="utf-8") as f:
                self.__concept_graph_prompt_template = "\n".join(f.readlines())
            with open(concept_prompt_file, "r", encoding="utf-8") as f:
                self.__concept_prompt_template = "\n".join(f.readlines())
            self.client = openai.OpenAI() if USE_OPENAI else ReplicateAPI()
            
        def __get_prompt(self, tutor_plan, main_concept, concept_list, wiki_data, concept_name, error_msg):
            prompt = self.__concept_prompt_template
            prompt = prompt.replace(self.TUTOR_PLANNER_DELIMITER, tutor_plan).replace(self.CURR_ENV_MAIN_CONCEPT_DELIMITER, main_concept).replace(self.CURR_ENV_CONTEXT_CONCEPT_DELIMITER, concept_list,).replace(self.CONCEPT_NAME_DELIMITER, concept_name,).replace(self.CURR_ERROR_DELIMITER, error_msg)
            return prompt
        
        def request_concept_data_from_llm(self, env_main_concept, context_concept, wiki_data, concept_name, error_msg):
            """Requests the Concept information from an LLM.

            Args:
                env_main_concept (str): _description_ 
                env_concept_list (str): _description_
                concept_name (str): _description_

            Returns:
                str: LLM Output containing Concept Data
            """
            prompt = self.__get_prompt(self.__tutor_plan, env_main_concept, context_concept, wiki_data, concept_name, error_msg)
                
            # model = "gpt-3.5-turbo-1106" #if concept_name != env_main_concept else "gpt-4"
            if USE_OPENAI:
                model = "gpt-4-1106-preview"
                # model = "gpt-3.5-turbo-16k"
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Act as an intelligent AI Tutor.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=0.1,
                    max_tokens=2000,
                    top_p=0.9,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                return prompt, response.choices[0].message.content
            else:
                return prompt, self.client.get_output(prompt, " ")
            
        def request_concept_graph_from_llm(self, env_main_concept, temp = 0.5):
            """Requests the Concept information from an LLM.
            Args:
                env_main_concept (str): _description_ 

            Returns:
                str: LLM Output containing Concept Graph Data
            """
            prompt = self.__concept_graph_prompt_template.replace("$CURR_ENV.TUTOR_PLANNER$", self.__tutor_plan).replace("$CURR_ENV.MAIN_CONCEPT$", env_main_concept)
                
            # model = "gpt-3.5-turbo-1106" #if concept_name != env_main_concept else "gpt-4"
            if USE_OPENAI:
                model = "gpt-4-1106-preview"
                # model = "gpt-3.5-turbo-16k"
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Act as an intelligent AI Tutor.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=0.1,
                    max_tokens=2000,
                    top_p=0.9,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                return prompt, response.choices[0].message.content
            else:
                return prompt, self.client.get_output(prompt, " ")
            
    __CONCEPT_REGEX = re.compile(r'\`\`\`yaml([^\`]*)\`\`\`') # Matches any ```yaml ...```
    __GRAPH_REGEX = re.compile(r'\`\`\`plaintext([^\`]*)\`\`\`') # Matches any ```plaintext ... ```
    def __init__(self, main_concept:str, tutor_plan:str = "", max_threads=120): #TODO: Fix potential Resource lock issue 
        self.concept_llm_api = self.ConceptLLMAPI("AITutor_Backend/src/TutorUtils/Prompts/KnowledgePhase/Concepts/concept_graph_prompt", "AITutor_Backend/src/TutorUtils/Prompts/KnowledgePhase/Concepts/concept_prompt", tutor_plan=tutor_plan) # TODO: FIX
        self.main_concept = main_concept
        self.Concepts = []
    
    def generate_concept_graph(self, ):
        """
        Generates a Concept Graph from a LLM for the main concept
        """
        while True:
            prompt, llm_output = self.concept_llm_api.request_concept_graph_from_llm(self.main_concept, )
            try:
                with open("translation.txt", "a") as f:
                    f.write("TRANSLATION\n")
                regex_match = ConceptDatabase.__GRAPH_REGEX.findall(llm_output)
                assert regex_match, f"Error parsing LLM Output for Concept Graph. Ensure you properly used the Yaml Creation rules."
                # Extract the Yaml Data from the LLM Ouput\
                data = regex_match[0].replace("    ", "\t").strip()
                graph_data = ConceptDatabase.build_dict(data.split("\n"))
                assert graph_data, "Could not create concept from LLM Output, ensure you have properly entered the information and did not include any additional information outside of what's required."
                # Save the llm_output as training tata to file "training_data/concept/generation/":
                output_dir = "training_data/concept/graph_generation/"
                save_training_data(output_dir, prompt, llm_output)
                break
            except Exception as e:
                error_msg = str(e) # TODO: LOG 
                with open("translation_errors.txt", "a") as f:
                    f.write("TRANSLATION_ERROR\n")
        # Recursively add concepts to the graph
        def add_concepts_recursively(c, parent:Concept, cd:ConceptDatabase):
            concept = Concept(c['concept'], parent)
            cd.Concepts.append(concept)
            if parent is not None: parent.refs.append(concept)
            for c_ref in c['refs']:
                add_concepts_recursively(c_ref, concept, cd)

        for concept in graph_data[0]:
            add_concepts_recursively(concept, None, self)

        if DEBUG: print("Concept Graph:\n", self.get_concept_graph_str())

    def get_concept_list_str(self,):
        return "\n".join([f"\t- \"{concept.name}\"" for concept in self.Concepts],) if self.Concepts else "The Concept List is Empty."
    
    def get_concept_graph_str(self,):
        origin_nodes = [c for c in self.Concepts if c.parent == None]
        c_graph = ""
        def rec_gen_c_graph(node:Concept, tab=1):
            return f"{node.name}\n\n" + "\n".join(["\t"*tab+rec_gen_c_graph(c, tab+1) for c in node.refs]) if node != None else ""
        for origin_node in origin_nodes:
            c_graph+=rec_gen_c_graph(origin_node)
        return c_graph
            
    def get_concept(self, concept_name:str): # We use edit distance of 4 as a max
        if not concept_name: return None
        concept = [(edit_distance(concept_name.lower(), concept.name.lower()), concept) for concept in self.Concepts if edit_distance(concept_name.lower(), concept.name.lower()) < 4]
        if concept: concept.sort(key=lambda x: x[0])# Get the best match
        return concept[0][1] if concept else None
    
    def generate_concepts(self,):
        if DEBUG: print("Generating Concepts...")
        threads = []
        generation_lock = threading.Lock()
        for concept_ref in self.Concepts:
            thread = threading.Thread(target=self.generate_concept, args=(concept_ref, generation_lock))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

    def generate_concept(self, concept:"Concept", generation_lock:threading.Lock):
        """
        Generates a Concept from a LLM
        """
        if DEBUG: print("Generating Concept:", concept.name)
        def escape_backslashes_in_quotes(text):
            def replace_backslashes(match):
                return match.group(0).replace('\\', '\\\\')
            quoted_strings_regex = r'"([^"\\]*(?:\\.[^"\\]*)*)"'
            return re.sub(quoted_strings_regex, replace_backslashes, text)
        error_msg = "There is no current error detected in the parsing system."
        context_concept = concept.parent.name if concept.parent is not None else self.main_concept
        while True:
            prompt, llm_output = self.concept_llm_api.request_concept_data_from_llm(self.main_concept, context_concept,"Not available.", concept.name, error_msg)
            try:
                with open("translation.txt", "a") as f:
                    f.write("TRANSLATION\n")
                regex_match = ConceptDatabase.__CONCEPT_REGEX.findall(llm_output)
                assert regex_match, f"Error parsing LLM Output for Concept: {concept.name}. Ensure you properly used the Yaml Creation rules."
                # Extract the Yaml Data from the LLM Ouput
                parsed_data = escape_backslashes_in_quotes(regex_match[0])
                parsed_data = yaml.safe_load(parsed_data)
                # Type check
                assert isinstance(parsed_data, dict), "Root should be a dictionary"
                # Key check
                assert 'Concept' in parsed_data, "Concept field is missing in the YAML data"
                assert all(key in parsed_data['Concept'] for key in ['name', 'definition',]), "Some required keys are missing in Concept"
                # Extract info from LLM Output
                c_def = parsed_data["Concept"]["definition"]
                c_latex =  parsed_data["Concept"].get("latex_code", None)
                if c_latex and (c_latex.lower() == "none" or c_latex.lower() == "null"): c_latex = ""
                concept.set_definition(c_def)
                concept.latex = c_latex
                assert concept.definition, "Could not create concept from LLM Output, ensure you have properly entered the information and did not include any additional information outside of what's required."
                # Save the llm_output as training tata to file "training_data/concept/generation/":
                output_dir = "training_data/concept/generation/"
                save_training_data(output_dir, prompt, llm_output)
                break
            except Exception as e:
                error_msg = str(e)
                with open("translation_errors.txt", "a") as f:
                    f.write("TRANSLATION_ERROR\n")
                
    @staticmethod                
    def from_sql(main_concept, tutor_plan, concepts):
        """creates a ConceptDatabase from sql data.

        Args:
            main_concept (str): _description_
            tutor_plan (str): plan from tutor
            concepts (List[str, str, str, str, str]): List[(concept_name, parent.name, concept_def, concept_latex, refs_str)] 

        Returns:
            _type_: ConceptDatabase
        """
        cd = ConceptDatabase(main_concept, tutor_plan, )
        split_refs = lambda x: x.split("[SEP]") if x is not None else x
        cd.Concepts = [Concept.from_sql(cpt[0], cpt[1], cpt[2], cpt[3], split_refs(cpt[4]),) for cpt in concepts]
        for concept in cd.Concepts: # Update/Populate references from the temporary references
            concept.parent = cd.get_concept(concept.parent)
            if concept.refs:
                concept.refs = [cd.get_concept(ref) for ref in concept.refs]
                concept.refs = [ref for ref in concept.refs if ref is not None] # Filter None-references
            else:
                concept.refs = []
        return cd
    
    def to_sql(self, ):
        """
        Returns:
            main_concept (str): 
            concepts (List[str, str, str]): List[(concept_name, concept_def, concept_latex)]
        """
        return (self.main_concept, [c.to_sql() for c in self.Concepts])
        
        
     
class Concept(JSONSerializable):
    def __init__(self, name:str, parent):
         self.name = name
         self.parent = parent #TODO: Implement Parent in Database 
         self.definition = ""
         self.latex = ""
         self.refs = []

    def __repr__(self) -> str:
        return f"Concept(name: {self.name}) <{self.__hash__()}>"
    
    def format_json(self,):
        map_concept = lambda c: c.name if isinstance(c, Concept) else c
        return {
            "name": self.name,
            "definition": " ".join([map_concept(c) for c in self.definition]),
            "latex": self.latex
        }

    def set_definition(self, definition):
        self.definition = definition

    def set_latex(self, latex):
        self.latex = latex

    @staticmethod
    def from_sql(concept_name, parent, concept_def, concept_latex, refs):
        concept = Concept(concept_name, parent)
        concept.refs = refs
        concept.set_definition(concept_def)
        concept.set_latex(concept_latex)
        return concept
    def to_sql(self,) -> Tuple[str, str, str]:
        """Returns the state of Concept

        Returns:
            Tuple[str, str, str, str, str]: (concept_name, parent.name, concept_def, concept_latex, refs_str) 
        """

        refs = "[SEP]".join([ref.name for ref in self.refs if isinstance(ref, Concept)]) if self.refs else None
        parent_name = self.parent.name if self.parent else None
        return (self.name, parent_name, self.definition, self.latex, refs)
  