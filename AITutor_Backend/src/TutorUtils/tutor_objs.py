import json
from typing import List, Tuple, Union
import re
from enum import IntEnum
import threading
import openai

from AITutor_Backend.src.BackendUtils.sql_serialize import SQLSerializable
from AITutor_Backend.src.BackendUtils.json_serialize import JSONSerializable
from AITutor_Backend.src.BackendUtils.env_serialize import EnvSerializable

from AITutor_Backend.src.TutorUtils.concepts import *
from AITutor_Backend.src.TutorUtils.notebank import *

from AITutor_Backend.src.DataUtils.file_utils import save_training_data
from enum import IntEnum
from typing import List, Tuple

import os
DEBUG = bool(os.environ.get("DEBUG", 0))

class TutorObjLLMAPI:
    # CHAPTERS
    CONCEPT_GRAPH_DELIMITER = "$ENV.CONCEPTS$"
    NOTEBANK_STATE_DELIMITER = "$ENV.NOTEBANK_STATE$"
    MAIN_CONCEPT_DELIMITER = "$ENV.MAIN_CONCEPT$"
    
    ### LESSONS
    PREV_CHAPTER_DELIMITER = "$ENV.PREV_CHAPTER$"
    CURR_CHAPTER_DELIMITER = "$ENV.CURR_CHAPTER$"
    

    def __init__(self, chapter_plan_prompt_file, chapter_obj_prompt_file, lesson_plan_prompt_file, lesson_obj_prompt_file):
        self.client = openai.OpenAI() if USE_OPENAI else ReplicateAPI() #TODO: Add support for llamma.cpp
        with open(chapter_plan_prompt_file, "r") as f:
            self._chapter_plan_prompt = f.read()
        with open(chapter_obj_prompt_file, "r") as f:
            self._chapter_obj_prompt = f.read()
        with open(lesson_plan_prompt_file, "r") as f:
            self._lesson_plan_prompt = f.read()
        with open(lesson_obj_prompt_file, "r") as f:
            self._lesson_obj_prompt = f.read()
            
    def request_output_from_llm(self, prompt, model: str, max_length = 3500, temp=0.5):
        """Requests data from an LLM.

        Args:
            prompt: (str) - string to get passed to the model
            model: (str) - name of the model to use
            max_length: (int) - max number of tokens for the request
            temp: (float) - temperature for LLM to use in the request

        Returns:
            str: llm output
        """
        if USE_OPENAI:
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
                temperature=temp,
                max_tokens=max_length,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )

            return response.choices[0].message.content
        else:
            return self.client.get_output(prompt, " ")
        
    def _load_prompt(self, prompt_template, state_dict):
        prompt_string = prompt_template
        # Replace Values in Prompt:
        for k, v in state_dict.items():
            prompt_string = prompt_string.replace(k, v)

        # Return Prompt:
        return prompt_string
    
    def prompt_lesson_plan(self, notebank_state, main_concept, prev_chapter, curr_chapter):
        return self._load_prompt(self._chapter_plan_prompt, {
            TutorObjLLMAPI.NOTEBANK_STATE_DELIMITER: notebank_state,
            TutorObjLLMAPI.MAIN_CONCEPT_DELIMITER: main_concept,
            TutorObjLLMAPI.PREV_CHAPTER_DELIMITER: prev_chapter,
            TutorObjLLMAPI.CURR_CHAPTER_DELIMITER: curr_chapter,

        })
    
    def prompt_chapter_plan(self, notebank_state, concept_graph_str):
        return self._load_prompt(self._lesson_plan_prompt, {
            TutorObjLLMAPI.NOTEBANK_STATE_DELIMITER: notebank_state,
            TutorObjLLMAPI.CONCEPT_GRAPH_DELIMITER: concept_graph_str,
        })
    

class TutorObjManager:
    def __init__(self, notebank:NoteBank, concept_database:ConceptDatabase,):
        super(TutorObjManager, self).__init__()
        self.llm_api = TutorObjLLMAPI("AITutor_Backend/src/TutorUtils/Prompts/KnowledgePhase/TutorObjs/chapter_plan_prompt", "AITutor_Backend/src/TutorUtils/Prompts/KnowledgePhase/TutorObjs/chapter_obj_prompt", "AITutor_Backend/src/TutorUtils/Prompts/KnowledgePhase/TutorObjs/lesson_plan_prompt", "AITutor_Backend/src/TutorUtils/Prompts/KnowledgePhase/TutorObjs/lesson_obj_prompt")
        self.Chapters = []
        self.num_chapters = 0
        self.cd = concept_database
        self.notebank = notebank
        self.current_chapter_idx = -1
        self.initialized = False
    
    def generate_chapters(self, ):
        """

        """
        while True:
            slides_prompt = self.llm_api.prompt_chapter_plan(self.notebank.env_string(), self.cd.get_concept_graph_str())
            llm_output = self.llm_api.request_output_from_llm(slides_prompt, "gpt-4-1106-preview")
            slides_json_conversion = self.llm_api.request_output_from_llm(self.llm_api.__chapter_obj_prompt, )
            success, chapters = Chapter.create_chapters_from_JSON(llm_output)
            if not success: continue
            self.Chapters = chapters
            self.num_chapters = len(self.Chapters)
            break
        self.initialized = True

    def initialize_chapter(self, chapter_idx,):
        if not 0 <= chapter_idx < self.num_chapters:
            raise IndexError("Cannot Access Out-Of-Bounds Chapters.")
        

class Chapter(JSONSerializable, SQLSerializable, EnvSerializable):
    JSON_REGEX = re.compile(r'\`\`\`json([^\`]*)\`\`\`')
    def __init__(self, title, overview, outcomes, concepts):
        self.Lessons = []
        self.title = title
        self.overview = overview
        self.outcomes = outcomes
        self.concepts = concepts
        self.initialized = False
    
    @staticmethod
    def create_chapters_from_JSON(llm_output, cd:ConceptDatabase):
        try: 
            regex_match = Chapter.JSON_REGEX.findall(llm_output)
            if regex_match:
                regex_match = regex_match[0].replace("```json", "").replace("```", "").strip()
            chapter_data = regex_match if regex_match else llm_output
            chapter_data = json.loads(chapter_data)
            chapters = [Chapter(ch['title'], ch["overview"], ch['outcomes'], [c for c in [cd.get_concept(concept) for concept in ch['concepts']] if c is not None]) for ch in chapters]
            return True, chapters
        except json.JSONDecodeError:
            return False, "Could not decode the object"

class Lesson(JSONSerializable, SQLSerializable, EnvSerializable,):
    def __init__(self, title, overview, objectives, concepts):
        self.Slides = []
        self.Homework = []
        self.title = title
        self. overview = overview
        self.objectives = objectives
        self.concepts = concepts