import json
import os
import re
import threading
from enum import IntEnum
from typing import List, Tuple, Union


from AITutor_Backend.src.BackendUtils.extractors import JSONExtractor
from AITutor_Backend.src.BackendUtils.env_serialize import EnvSerializable
from AITutor_Backend.src.BackendUtils.json_serialize import JSONSerializable
from AITutor_Backend.src.BackendUtils.sql_serialize import SQLSerializable
from AITutor_Backend.src.BackendUtils.completable import Completable
from AITutor_Backend.src.DataUtils.file_utils import save_training_data, json_to_string
from AITutor_Backend.src.TutorUtils.concepts import *
from AITutor_Backend.src.TutorUtils.notebank import *

from AITutor_Backend.src.TutorUtils.Modules.slides import SlidePlanner, Slide
from AITutor_Backend.src.TutorUtils.Modules.questions import QuestionSuite, Question

from AITutor_Backend.src.BackendUtils.llm_client import LLM

from AITutor_Backend.src.PromptUtils.prompt_template import PromptTemplate
from AITutor_Backend.src.PromptUtils.prompt_utils import (
    Message,
    Conversation,
    AI_TUTOR_MSG,
)


DEBUG = bool(os.environ.get("DEBUG", 0))


class TutorObjPrompts:
    # CHAPTERS
    CONCEPT_GRAPH_DELIMITER = "$ENV.CONCEPTS$"
    NOTEBANK_STATE_DELIMITER = "$ENV.NOTEBANK_STATE$"
    MAIN_CONCEPT_DELIMITER = "$ENV.MAIN_CONCEPT$"

    ### LESSONS
    PREV_CHAPTER_DELIMITER = "$ENV.PREV_CHAPTER$"
    CURR_CHAPTER_DELIMITER = "$ENV.CURR_CHAPTER$"

    def __init__(
        self
    ):
        # Load chapter plan prompt
        self._chapter_plan_prompt = PromptTemplate.from_config(
            "@planChapters",
            {
                "notebank": TutorObjPrompts.NOTEBANK_STATE_DELIMITER,
                "concepts": TutorObjPrompts.CONCEPT_GRAPH_DELIMITER,
            },
        )
        """vars: notebank, concept_graph"""

        # Load lesson plan prompt
        self._lesson_plan_prompt = PromptTemplate.from_config(
            "@planLessons",
            {
                "notebank": TutorObjPrompts.NOTEBANK_STATE_DELIMITER,
                "concepts": TutorObjPrompts.CONCEPT_GRAPH_DELIMITER,
                "prev_chapter": TutorObjPrompts.PREV_CHAPTER_DELIMITER,
                "curr_chapter": TutorObjPrompts.CURR_CHAPTER_DELIMITER,
            },
        )
        """vars: notebank, main_concept, prev_chapter, curr_chapter"""

        # Load chapter obj prompt
        self._chapter_obj_prompt = PromptTemplate.from_config(
            "@objConvertChapters",
            {
                "chapters": "$CHAPTERS$",
            },
        )
        """vars: chapters"""

        # Load chapter obj prompt
        self._lesson_obj_prompt = PromptTemplate.from_config(
            "@objConvertLessons",
            {
                "lessons": "$LESSONS$",
            },
        )
        """vars: lessons"""

    def _load_prompt(self, prompt_template, state_dict):
        prompt_string = prompt_template
        # Replace Values in Prompt:
        for k, v in state_dict.items():
            prompt_string = prompt_string.replace(k, v)

        # Return Prompt:
        return prompt_string

    def prompt_lesson_plan(
        self, notebank_state, concept_graph_str, prev_chapter, curr_chapter
    ):
        return self._lesson_plan_prompt.replace(
            notebank=notebank_state,
            concepts=concept_graph_str,
            prev_chapter=prev_chapter,
            curr_chapter=curr_chapter,
        )

    def prompt_chapter_plan(self, notebank_state, concept_graph_str):
        return self._chapter_plan_prompt.replace(
            notebank=notebank_state, concepts=concept_graph_str
        )


class TutorObjManager(SQLSerializable, JSONSerializable, Completable):
    def __init__(
        self,
        notebank: NoteBank,
        concept_database: ConceptDatabase,
    ):
        super(TutorObjManager, self).__init__()
        self.llm_prompts = TutorObjPrompts()
        self.Chapters = []
        self.num_chapters = 0
        self.cd = concept_database
        self.notebank = notebank

        self.curr_chapter_idx = -1
        self.curr_lesson_idx = -1

    @property
    def initialized(self):
        return self.num_lessons > 0

    @staticmethod
    def from_sql(
        notebank: NoteBank,
        cd: ConceptDatabase,
        chapters_data: List[tuple],
        curr_chapter_idx: int,
        current_lesson_idx: int,
    ):
        raise NotImplementedError()

    def to_sql(self):
        raise NotImplementedError()

    def format_json(self):
        return {
            "Chapters": [chapter.format_json() for chapter in self.Chapters],
            "curr_chapter_idx": self.curr_chapter_idx,
            "curr_lesson_idx": self.curr_lesson_idx,
        }

    def generate_chapters(
        self,
    ):
        """Generation Process: \n
        1. (concept_graph, notebank) -> Generate N Chapters
        2. Convert into N Chapter Objects
        """
        notebank_context = self.notebank.generate_context_summary(
            query=f"Creating Chapters teaching about {self.cd.main_concept}",
            objective=f"Generate a summary of the provided context for creating a a set of Chapters teaching about {self.cd.main_concept} and its concepts: \n{self.cd.get_concept_graph_str()}",
        )
        while True:
            # Load slides prompt
            chapters_prompt = self.llm_prompts.prompt_chapter_plan(
                notebank_context, self.cd.get_concept_graph_str()
            )

            # Create msgs
            messages = Conversation.from_message_list(
                [AI_TUTOR_MSG, Message(role="user", content=chapters_prompt)]
            )

            # Get the output
            llm_output = LLM("@planChapters").chat_completion(messages=messages)

            # Get the conversion
            conversion_messages = Conversation.from_message_list(
                [
                    AI_TUTOR_MSG,
                    Message(
                        role="user",
                        content=self.llm_prompts._chapter_obj_prompt.replace(
                            chapters=llm_output
                        ),
                    ),
                ]
            )

            slides_json_conversion = LLM("@objConvertChapters").chat_completion(
                messages=conversion_messages
            )

            # Retrieve the chapter objects:
            success, chapters = Chapter.create_chapters_from_JSON(
                slides_json_conversion, self.cd
            )

            if not success:
                continue

            assert len(chapters) > 0, "No chapters created for this Module."

            # Save as training Data:
            output_dir = "training_data/chapter/chapter_plan/"
            save_training_data(
                output_dir, json_to_string(messages.format_json()), llm_output
            )

            output_dir = "training_data/chapter/chapter_obj/"
            save_training_data(
                output_dir,
                json_to_string(conversion_messages.format_json()),
                slides_json_conversion,
            )

            # Populate data:
            self.Chapters = chapters
            self.num_chapters = len(self.Chapters)
            break

    def initialize_chapter(
        self,
        chapter_idx: int,
    ):
        """Initializes a chapter at index `chapter_idx`"""
        if not 0 <= chapter_idx < self.num_chapters:
            raise IndexError("Cannot Access Out-Of-Bounds Chapters.")

        self._generate_modules(chapter_idx)

    def _generate_modules(self, chapter_idx: int):
        """
        Generates a Modules from the provided Chapter Context.

        Generation Process: \n
            1. (prev_chapter, curr_chapter, main_concept, notebank) -> Generate K Lessons
            2. Convert into Lessons Objects
            3. Generate the content for Lessons[0]
        """
        prev_chapter = (
            self.Chapters[chapter_idx - 1].env_string()
            if chapter_idx > 0
            else " There is no previous Chapter."
        )
        curr_chapter = self.Chapters[chapter_idx].env_string()
        notebank_context = self.notebank.generate_context_summary(
            query=f"Creating Lessons for Chapter:\n{curr_chapter}",
            objective=f"Generate a summary of the provided context for creating a a set of Lessons for Chapter:\n{curr_chapter}",
        )
        while True:
            # Load lesson prompt
            lessons_prompt = self.llm_prompts.prompt_lesson_plan(
                notebank_context,
                self.cd.get_concept_graph_str(),
                prev_chapter,
                curr_chapter,
            )

            # Create msgs
            messages = Conversation.from_message_list(
                [AI_TUTOR_MSG, Message(role="user", content=lessons_prompt)]
            )

            # Get the output
            llm_output = LLM("@planLessons").chat_completion(messages=messages)

            lessons_conversion_messages = Conversation.from_message_list(
                [
                    AI_TUTOR_MSG,
                    Message(
                        role="user",
                        content=self.llm_prompts._lesson_obj_prompt.replace(
                            lessons=llm_output
                        ),
                    ),
                ]
            )

            # Get the conversion
            lessons_json_conversion_output = LLM("@objConvertLessons").chat_completion(
                messages=lessons_conversion_messages
            )

            # Retrieve the lessons objects:
            success, lessons = Lesson.create_lessons_from_JSON(
                lessons_json_conversion_output, self.notebank, self.cd
            )

            if not success:
                continue

            # Save Output to local directory:
            output_dir = "training_data/lesson/lesson_plan/"
            save_training_data(
                output_dir, json_to_string(messages.format_json()), llm_output
            )

            output_dir = "training_data/lesson/lesson_obj/"
            save_training_data(
                output_dir,
                json_to_string(lessons_conversion_messages.format_json()),
                lessons_json_conversion_output,
            )

            self.Modules = lessons
            self.num_chapters = len(self.Chapters)

            # Initialize first lesson, populate chapter's lesson data:
            if len(lessons) > 0:
                chapter = self.Chapters[chapter_idx]
                lessons[0].initialize_lesson(chapter.env_string())
                self.Chapters[chapter_idx].curr_lesson_idx = 0
                self.Chapters[chapter_idx].Lessons = lessons
                self.Chapters[chapter_idx].num_lessons = len(lessons)
                break  # Finished Generating the Chapter

    def get_chapter(self, chapter_idx: int) -> "Chapter":
        """Initializes a chapter at index `chapter_idx`"""
        if not 0 <= chapter_idx < self.num_chapters:
            raise IndexError("Cannot Access Out-Of-Bounds Chapters.")

        return self.Chapters[chapter_idx]


class Chapter(JSONSerializable, SQLSerializable, EnvSerializable):
    JSON_REGEX = re.compile(r"\`\`\`json([^\`]*)\`\`\`")

    def __init__(self, title, overview, outcomes, concepts):
        self.Lessons = []  # TODO: Change to be abstract class Module
        self.num_lessons = 0
        self.curr_lesson_idx = -1
        self.title = title
        self.overview = overview
        self.outcomes = outcomes
        self.concepts = concepts

    @property
    def initialized(self):
        return self.num_lessons > 0

    @staticmethod
    def create_chapters_from_JSON(llm_output, cd: ConceptDatabase):
        try:
            chapter_data = JSONExtractor.extract(llm_output)
            print("[DBG] Chapter Data:", chapter_data)
            chapters = [
                Chapter(
                    ch["title"],
                    ch["overview"],
                    (
                        ch["outcomes"]
                        if isinstance(ch["outcomes"], list)
                        else [ch["outcomes"]]
                    ),
                    [
                        c
                        for c in [cd.get_concept(concept) for concept in ch["concepts"]]
                        if c is not None
                    ],
                )
                for ch in chapter_data.get("Chapters", list(chapter_data.keys())[0])
            ]
            return True, chapters
        except (json.JSONDecodeError, KeyError, TypeError):
            return False, "Could not decode the object"

    def env_string(self):
        return (
            f"""### Chapter Title: {self.title}"""
            + "\n - **Overview:** {self.overview}\n"
            + "- **Outcomes:**\n"
            + "\n".join([f"\t{i}. {o}" for i, o in enumerate(self.outcomes)])
            + "\n"
            + " - **Concepts:**\n"
            + "\n".join([f"\t - {c.name}" for c in self.concepts])
        )


class Lesson(
    JSONSerializable,
    SQLSerializable,
    EnvSerializable,
):
    def __init__(
        self, title, overview, objectives, concepts, notebank, _ConceptDatabase
    ):
        self.Slides = None
        self.curr_slide_idx = -1

        self.Questions = None
        self.curr_question_idx = -1

        self.initialized = False

        self.title = title
        self.overview = overview
        self.objectives = objectives
        self.concepts = concepts
        self.initialized = False
        self.notebank = notebank
        self.cd = _ConceptDatabase

    def initialize_lesson(self, chapter_plan: str):
        """
        Generates a Lesson from the provided Chapter Context.

        Generation Process: \n
            1. (chapter, lesson) -> Slides
            2. (chapter, lesson) -> Questions
        """
        import threading

        # Generate Slides asynchronously
        def generate_slides():
            self.Slides = SlidePlanner(self.notebank, self.cd)
            self.Slides.generate_slide_deque(chapter_plan, self.env_string())

        # Generate Questions asynchronously
        def generate_questions():
            self.Questions = QuestionSuite(self.notebank, self.cd)
            self.Questions.generate_question_suite(chapter_plan, self.env_string())

        # Create and start threads
        slide_thread = threading.Thread(target=generate_slides)
        question_thread = threading.Thread(target=generate_questions)

        slide_thread.start()
        question_thread.start()

        # Wait for both threads to complete
        slide_thread.join()
        question_thread.join()

        self.initialized = True

    def format_json(self):
        data = {
            # "slides": self.Slides.env_string(),
            "curr_slide_idx": self.curr_slide_idx,
            # "questions": self.Questions.env_string(),
            "curr_question_idx": self.curr_question_idx,
        }
        data.update(self.Slides.env_string())

        return data

    @classmethod
    def create_lessons_from_JSON(
        cls, llm_output, notebank: NoteBank, cd: ConceptDatabase
    ) -> Tuple[bool, List["Lesson"]]:
        try:
            lesson_data = JSONExtractor.extract(llm_output)

            # Try to load each individual lesson
            lessons = [
                cls(
                    l["title"],
                    l["overview"],
                    (
                        l["objectives"]
                        if isinstance(l["objectives"], list)
                        else [l["objectives"]]
                    ),
                    [
                        c
                        for c in [cd.get_concept(concept) for concept in l["concepts"]]
                        if c is not None
                    ],
                    notebank,
                    cd,
                )
                for l in lesson_data["Lessons"]
            ]

            return True, lessons

        except json.JSONDecodeError:
            return False, "Could not decode the object"

        except Exception as err:
            pass

    def env_string(self):
        return (
            f"""### Lesson Title: {self.title}\n - **Overview:** {self.overview}\n - **Objectives:**\n"""
            + "\n".join([f"\t{i}. {o}" for i, o in enumerate(self.objectives)])
            + "\n"
            + " - **Concepts:**\n"
            + "\n".join([f"\t - {c.name}: {c.definition}" for c in self.concepts])
        )
