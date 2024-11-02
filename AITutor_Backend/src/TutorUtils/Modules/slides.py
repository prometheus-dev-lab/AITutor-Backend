import json
import os
import re
import threading
from enum import IntEnum
from typing import List, Tuple, Union, Optional

import openai
import uuid

from AITutor_Backend.src.BackendUtils.env_serialize import EnvSerializable
from AITutor_Backend.src.BackendUtils.extractors import JSONExtractor
from AITutor_Backend.src.BackendUtils.json_serialize import JSONSerializable
from AITutor_Backend.src.DataUtils.file_utils import save_training_data
from AITutor_Backend.src.TutorUtils.concepts import *
from AITutor_Backend.src.BackendUtils.llm_client import LLM

from AITutor_Backend.src.PromptUtils.prompt_template import PromptTemplate
from AITutor_Backend.src.PromptUtils.prompt_utils import (
    Message,
    Conversation,
    AI_TUTOR_MSG,
)

DEBUG = bool(os.environ.get("DEBUG", 0))

FORMAT_CONCEPTS = lambda concepts: "\n".join(
    [f"Concept: {c.name}\nDefinition: {c.to_tokenized_def()}\n" for c in concepts]
)


class Slide(JSONSerializable, SQLSerializable, EnvSerializable):
    def __init__(
        self,
        title: str,
        content: str,
        img_caption: Optional[str],
        ltx_codes: str,
        concepts: List[Concept],
        id: uuid.UUID = None,
    ):
        self.id = id or uuid.uuid4()
        self.title = title
        self.content = content
        self.img_caption = img_caption
        self.ltx_codes = ltx_codes
        self.concepts = concepts

    def format_json(self):
        return {
            "id": str(self.id),
            "title": self.title,
            "content": self.content,
            "img_caption": self.img_caption,
            "ltx_codes": self.ltx_codes,
            "concepts": [concept.name for concept in self.concepts],
        }

    def env_string(self):
        concepts_str = "\n".join(
            [f"\t - {c.name}: {c.definition}" for c in self.concepts]
        )
        return (
            f"### Slide: {self.title}\n"
            f"**Content**: {self.content}\n"
            f"**Image Caption**: {self.img_caption or 'N/A'}\n"
            f"**Concepts**:\n{concepts_str}"
        )

    @staticmethod
    def from_dict(data: dict, concept_database: ConceptDatabase):
        return Slide(
            id=uuid.UUID(data["id"]),
            title=data["title"],
            content=data["content"],
            img_caption=data["img_caption"],
            ltx_codes=data["ltx_codes"],
            concepts=[concept_database.get_concept(c) for c in data["concepts"]],
        )
    
    @staticmethod
    def create_slides_from_JSON(llm_output: str, concept_database: ConceptDatabase):
        try:
            slides = []
            slide_obj_data = JSONExtractor.extract(llm_output)["slides"]
            for slide in slide_obj_data:
                # Perform Assertions:
                assert (
                    "title" in slide
                    and "content" in slide
                    and "concepts" in slide
                ), "Invalid keys found int slide."
                assert isinstance(slide["title"], str)
                assert isinstance(slide["content"], str)
                assert (
                    isinstance(slide["concepts"], list)
                    or slide["concepts"] is None
                ), "Invalid keys found int slide."

                # Ensure the Concepts is of type list
                if slide["concepts"] is None:
                    slide["concepts"] = []

                # Retrieve Concepts:
                slide_concepts = [
                    concept_database.get_concept(c)
                    for c in slide["concepts"]
                ]

                slides.append(
                    Slide(
                        slide["title"],
                        slide["content"],
                        slide.get("img_caption", None),
                        "", # LTX Codes
                        [c for c in slide_concepts if c is not None],
                    )
                )
            assert len(slides) > 0, "No slides were created."
            
        except Exception as e:
            print("[ERROR] Failed to create slides from JSON")
            print(e)
            return False, None
        
        return True, slides


class SlidePlanner(JSONSerializable, SQLSerializable):
    class SlidePlanPrompts:
        LESSON_PLAN_DELIMITER = "$ENV.LESSON_PLAN$"  # Environment for the notebankd
        CONCEPTS_DELIMITER = "$ENV.CONCEPTS$"  # Environment for the chat history
        NOTEBANK_STATE_DELIMITER = "$ENV.NOTEBANK_STATE$"
        CURR_ENV_CONEPT_LIST = "$ENV.CONCEPT_LIST$"
        CURR_ENV_SLIDE_PLAN = "$ENV.SLIDE_PLAN$"
        CURR_ENV_CONEPT_DATA_DELIMITER = "$ENV.CONCEPT_DATA$"
        CURR_ENV_CHAPTER_DELIMITER = "$ENV.CHAPTER_PLAN$"
        CURR_ENV_LESSON_DELIMITER = "$ENV.LESSON_PLAN$"

        def __init__(
            self,
        ):  # Continue obj

            self._slide_plan_prompt = PromptTemplate.from_config(
                "@planSlides",
                {
                    "notebank": SlidePlanner.SlidePlanPrompts.NOTEBANK_STATE_DELIMITER,
                    "concepts": SlidePlanner.SlidePlanPrompts.CONCEPTS_DELIMITER,
                    "chapter": SlidePlanner.SlidePlanPrompts.CURR_ENV_CHAPTER_DELIMITER,
                    "lesson": SlidePlanner.SlidePlanPrompts.CURR_ENV_LESSON_DELIMITER,
                },
            )
            """vars: notebank, concepts, chapter, lesson"""

            self._slide_obj_prompt = PromptTemplate.from_config(
                "@objConvertSlides", {"slides": "$SLIDES$"}
            )
            """vars: slides"""

            self._slide_dialogue_prompt = PromptTemplate.from_config(
                "@dialogueSlides",
                {
                    "slide_context": "$SLIDE_CONTEXT$",
                    "title": "$TITLE$",
                    "content": "$CONTENT$",
                    "img_caption": "$IMG_CAPTION$",
                },
            )
            """vars: slide_context, title, content, img_caption"""

    def __init__(self, Notebank, _ConceptDatabase: ConceptDatabase):
        self.SlidePlans = []
        self.Slides = []
        self.Notebank = Notebank
        self.ConceptDatabase = _ConceptDatabase
        self.num_slides = 0
        self.llm_prompts = SlidePlanner.SlidePlanPrompts()

    def to_sql(self):
        return {
            "current_obj_idx": self.current_obj_idx,
            "num_slides": self.num_slides,
            "slides": [slide.format_json() for slide in self.slides],
        }

    @staticmethod
    def from_sql(data: dict, notebank, concept_database):
        slides = [
            Slide.from_dict(slide_data, concept_database)
            for slide_data in data["slides_data"]
        ]

        slide_planner = SlidePlanner(
            notebank,
            concept_database,
        )
        slide_planner.slides = slides
        slide_planner.current_obj_idx = data["current_obj_idx"]
        slide_planner.num_slides = data["num_slides"]
        return slide_planner

    def format_json(
        self,
    ):
        return {
            "slides": [slide.format_json() for slide in self.Slides],
            "num_slides": self.num_slides,
        }

    def get_object(self, idx):
        """
        Returns Slide Object iff idx is a valid Slide Object index. Else, AssertionError
        """
        assert (
            0 <= idx < self.num_slides
        ), "Cannot access Slide Object Array Out of Bounds"
        return self.Slides[idx]

    def generate_slides(self, chapter_data: str, lesson_data: str) -> None:
        """
        Generates a List of Slides from the provided Chapter Context and Lesson Context.

        Generation Process: \n
            1. (chapter_plan, lesson_plan) -> Generate N Slides
            2. (chapter_plan, lesson_plan, (sliding) Slide Window at idx I) -> Slide Dialogue, for each Ith Slide Generated prior
        """
        if DEBUG:
            print(f"Generating Slides for {self.ConceptDatabase.main_concept}")
        notebank_state = self.Notebank.generate_context_summary(
            query=f"Creating questions for Chapter {chapter_data}, Lesson: {lesson_data}",
            objective=f"Generate a summary of the provided context for creating questions for Chapter {chapter_data}, Lesson: {lesson_data}",
        )
        while True:
            # Prepare input for LLM
            slides_prompt = self.llm_prompts._slide_plan_prompt.replace(
                notebank=notebank_state,
                concepts=self.ConceptDatabase.get_concept_graph_str(),
                chapter=chapter_data,
                lesson=lesson_data,
            )

            messages = Conversation.from_message_list(
                [AI_TUTOR_MSG, Message(role="user", content=slides_prompt)]
            )

            # Request output from LLM
            llm_output = LLM("@planSlides").chat_completion(messages)

            output_dir = "training_data/slides/slide_plan/"
            save_training_data(output_dir, slides_prompt, llm_output)

            # Convert:
            conversion_prompt = self.llm_prompts._slide_obj_prompt.replace(
                slides=llm_output
            )

            # Create msg with slides
            messages = Conversation.from_message_list(
                [AI_TUTOR_MSG, Message(role="user", content=conversion_prompt)]
            )
            error = "There currently is no error."
            for _ in range(5):  # 5 Retries, try to convert into an object
                try:
                    # with open("translation.txt", "a") as f:
                    #     f.write("TRANSLATION\n")
                    # Try running the obj translation
                    llm_output = LLM("@objConvertSlides").chat_completion(
                        messages, max_tokens=4000
                    )

                    # Extract, Type Check, and Initialize
                    success, slides = Slide.create_slides_from_JSON(llm_output, self.ConceptDatabase)
                    assert len(slides) > 1, "Failed to create slides from JSON."
                    if not success:
                        continue

                    output_dir = "training_data/slides/slide_obj/"
                    save_training_data(output_dir, conversion_prompt, llm_output)

                    # Update self data and return:
                    self.Slides = slides
                    self.num_slides = len(slides)
                    return

                except Exception as err:  # TODO: Fix error handling
                    error = str(err)
                    # with open("translation_errors.txt", "a") as f:
                    #     f.write("TRANSLATION_ERROR\n")

    def generate_slide_dialogue(self, slide: Slide, index, results, generation_lock):
        # Prepare Slide Dialogue Prompt
        slide_dialogue_prompt = self.llm_prompts._slide_dialogue_prompt.replace(
            self._generate_slide_window_context(index),
            title=slide.title,
            content=slide.content,
            img_caption=(slide.content if slide.content is not None else ""),
            # TODO: Add concept data to prompt
        )

        # Create Message:
        messages = Conversation.from_message_list(
            [AI_TUTOR_MSG, Message(role="user", content=slide_dialogue_prompt)]
        )

        while True:
            # with open("translation.txt", "a") as f:
            #     f.write("TRANSLATION\n")
            # Request output from LLM
            s_dialogue = LLM("@dialogueSlides").chat_completion(messages)

            # Attempt to extract then continue with retry on fail
            try:
                dialogue_data = JSONExtractor.extract(s_dialogue)["dialogue"]
            except:
                continue

            # Update the shared Attribute
            with generation_lock:
                slide.dialogue = dialogue_data

            output_dir = "training_data/slides/s_dialogue/"
            save_training_data(output_dir, slide_dialogue_prompt, s_dialogue)
            break
            # with open("translation_errors.txt", "a") as f:
            #     f.write("TRANSLATION_ERROR\n")

    def generate_slide_deque(
        self,
        chapter_plan: str,
        lesson_plan: str,
    ):
        """
        Generates a Slide Deque from the provided Chapter Context and Lesson Context.

        Generation Process: \n
            1. (chapter_plan, lesson_plan) -> Generate N Slides
            2. (chapter_plan, lesson_plan, (sliding) Slide Window at idx I) -> Slide Dialogue, for each Ith Slide Generated prior
        """
        # Initially generate the slide objects
        self.generate_slides(chapter_plan, lesson_plan)

        # Generate dialogues in parallel
        results = []
        threads = []
        generation_lock = threading.Lock()
        for idx, slide_plan_ref in enumerate(self.SlidePlans):
            thread = threading.Thread(
                target=self.generate_slide_dialogue,
                args=(slide_plan_ref, idx, results, generation_lock),
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

    def _generate_slide_window_context(self, idx, prev=1, upcoming=1):
        """
        Generates a slide window string from:
        slides[curr_idx - prev] : slides[curr_idx + upcoming]
        """
        newline = "\n"
        s = (
            """<Previous Slides>\n\n"""
            + f"""[{f'{newline+newline}'.join([
                    f"### Previous Slide: {newline}"+str(slide.env_string()) for i, slide in enumerate(
                        self.Slides[max(0, idx - prev):idx]
                    )
                ])
            }]"""
            + "\n\n</Previous Slides>"
        )
        if idx == self.num_slides - 1:
            s += "This is the Last Slide in the deque. Consider adding closing Remarks."

        s += (
            """\n<Upcoming Slides>"""
            + f"""[{f'{newline+newline}'.join([
                    f"Previous Slide: {newline}"+str(slide.env_string()) for i, slide in enumerate(
                        self.Slides[max(idx+1, self.num_slides-1):max(self.num_slides-1, idx + upcoming)]
                    )
                ])
            }]"""
            + "\n</Upcoming Slides>"
            if s != self.num_slides - 1
            else "There are no upcoming slides."
        )

        return s
