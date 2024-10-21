import json
import os
from enum import IntEnum
import threading
from typing import List, Tuple

from AITutor_Backend.src.BackendUtils.extractors import JSONExtractor
from AITutor_Backend.src.BackendUtils.env_serialize import EnvSerializable
from AITutor_Backend.src.BackendUtils.json_serialize import JSONSerializable
from AITutor_Backend.src.BackendUtils.sql_serialize import SQLSerializable
from AITutor_Backend.src.BackendUtils.completable import Completable
from AITutor_Backend.src.DataUtils.file_utils import save_training_data, json_to_string
from AITutor_Backend.src.BackendUtils.llm_client import LLM
from AITutor_Backend.src.PromptUtils.prompt_template import PromptTemplate
from AITutor_Backend.src.PromptUtils.prompt_utils import (
    Message,
    Conversation,
    AI_TUTOR_MSG,
)
from AITutor_Backend.src.TutorUtils.tutor_objs import Completable

DEBUG = bool(os.environ.get("DEBUG", 0))


class QuestionSuite(JSONSerializable, SQLSerializable, EnvSerializable, Completable):
    ALLOWED_LIBS = [
        ["numpy", "math", "sympy"],  # Math
        [
            "numpy",
            "math",
            "sympy",
            "collections",
            "itertools",
            "re",
            "heapq",
            "functools",
            "string",
            "torch",
            "nltk",
            "PIL",
            "cv2",
            "json",
            "enum",
            "typing",
        ],  # Python Programming
        None,
        None,
    ]

    class QuestionPrompts:
        def __init__(self):
            self.main_prompt = PromptTemplate.from_config(
                "@planQuestions",
                {
                    "main_concept": "$ENV.MAIN_CONCEPT$",
                    "context": "$CONTEXT$",
                    "chapter": "$ENV.CHAPTER$",
                    "lesson_plan": "$ENV.LESSON_PLAN$",
                    "error": "$ENV.CURR_ERROR$",
                },
            )
            self.question_to_obj_prompt = PromptTemplate.from_config(
                "@objConvertQuestions",
                {
                    "question_data": "$QUESTION_DATA$",
                },
            )

            self.question_data_prompt = PromptTemplate.from_config(
                "@questionData", {
                    "question_data": "$QUESTION_DATA$",
                    "subject_instructions": "$SUBJECT_INSTRUCTIONS$",
                    "type_instructions": "$TYPE_INSTRUCTIONS$",
                },
            )

            self.subject_prompts = self.load_subject_prompts()
            self.type_prompts = self.load_type_prompts()

        def load_subject_prompts(self):
            subject_prompts = {}
            for subject in Question.Subject:
                file_path = f"AITutor_Backend/src/TutorUtils/Prompts/KnowledgePhase/Questions/Subjects/{subject.name}_PROMPT"
                with open(file_path, "r") as f:
                    subject_prompts[subject] = f.read()
            return subject_prompts

        def load_type_prompts(self):
            type_prompts = {}
            for q_type in Question.Type:
                file_path = f"AITutor_Backend/src/TutorUtils/Prompts/KnowledgePhase/Questions/Types/{q_type.name}_PROMPT"
                with open(file_path, "r") as f:
                    type_prompts[q_type] = f.read()
            return type_prompts

    def __init__(self, Notebank, ConceptDatabase):
        super(QuestionSuite, self).__init__()
        self.current_obj_idx = -1
        self.Notebank = Notebank
        self.ConceptDatabase = ConceptDatabase
        self.Questions: List["Question"] = []
        self.num_questions = 0
        self.prompts = QuestionSuite.QuestionPrompts()

    def get_object(self, idx):
        """
        Returns Slide Object iff idx is a valid Slide Object index. Else, AssertionError
        """
        assert (
            0 <= idx < self.num_slides
        ), "Cannot access Slide Object Array Out of Bounds"
        return self.Questions[idx]

    def assert_question_data(self, q_data: dict) -> bool:
        assert isinstance(
            q_data, dict
        ), f"Error, Question Data is not a dictionary: {q_data}"

        # Assertions for each question type
        if q_data.get("type") == Question.Type.CODE_ENTRY:
            assert q_data.get(
                "code_prompt"
            ), f"Error, Code Entry Question is missing code prompt: {q_data}"

        elif q_data.get("type") == Question.Type.MULTIPLE_CHOICE:
            assert q_data.get(
                "options"
            ), f"Error, Multiple Choice Question is missing options: {q_data}"
            assert q_data.get(
                "correct_entry"
            ), f"Error, Multiple Choice Question is missing correct entry: {q_data}"

        elif q_data.get("type") == Question.Type.CALCULATION_ENTRY:
            assert q_data.get(
                "calculation_script"
            ), f"Error, Calculation Entry Question is missing correct answer: {q_data}"

        elif q_data.get("type") == Question.Type.TEXT_ENTRY:
            assert q_data.get(
                "text_prompt"
            ), f"Error, Text Entry Question is missing text prompt: {q_data}"

        else:
            raise ValueError(f"Error, Question Type is not supported: {q_data}")

        # Assertions for each subject
        if q_data.get("subject") == Question.Subject.MATH:
            assert q_data.get(
                "instructions"
            ), f"Error, Math Question is missing instructions: {q_data}"
            assert q_data.get(
                "latex_question"
            ), f"Error, Math Question is missing LaTeX representation: {q_data}"

        elif q_data.get("subject") == Question.Subject.CODE:
            assert q_data.get(
                "instructions"
            ), f"Error, Code Question is missing instructions: {q_data}"
            assert q_data.get(
                "boilerplate"
            ), f"Error, Code Question is missing boilerplate: {q_data}"
            assert q_data.get(
                "test_case_script"
            ), f"Error, Code Question is missing test case script: {q_data}"

        elif q_data.get("subject") == Question.Subject.LITERATURE:
            assert q_data.get(
                "instructions"
            ), f"Error, Literature Question is missing instructions: {q_data}"

        elif q_data.get("subject") == Question.Subject.CONCEPTUAL:
            assert q_data.get(
                "instructions"
            ), f"Error, Conceptual Question is missing instructions: {q_data}"

        return True

    def generate_question_data(
        self, q_obj: "Question", generation_lock: threading.Lock
    ):
        # Generate the question data
        prompt = self.prompts.question_to_obj_prompt.replace(
            question_data=q_obj.env_string(),
            subject_instructions=self.prompts.subject_prompts[q_obj.subject],
            type_instructions=self.prompts.type_prompts[q_obj.type],
        )
        messages = Conversation.from_message_list(
            [AI_TUTOR_MSG, Message(role="user", content=prompt)]
        )
        while True:
            try:
                # Request the question data from the LLM
                q_data = LLM("@objConvertQuestions").chat_completion(messages=messages)

                # Extract the question data from the LLM output
                q_data = JSONExtractor.extract(q_data)
                if isinstance(q_data, list):
                    q_data = q_data[0]

                # Check if the question data is valid
                _ = self.assert_question_data(q_data)
                break
            except Exception as e:
                error = f"Error while creating Question Data: {e}"
        with generation_lock:
            q_obj.data = q_data

    def generate_questions(self, chapter_plan: str, lesson_plan: str):
        if DEBUG:
            print(f"Generating Question Data for {self.ConceptDatabase.main_concept}")

        context_summary = self.Notebank.generate_context_summary(
            query=f"Creating questions for Chapter {chapter_plan}, Lesson: {lesson_plan}",
            objective=f"Generate a summary of the provided context for creating questions for Chapter {chapter_plan}, Lesson: {lesson_plan}",
        )

        error = "There is no current error."

        while True:
            try:
                # Generate the question plan
                prompt = self.prompts.main_prompt.replace(
                    main_concept=self.ConceptDatabase.main_concept,
                    context=context_summary,
                    chapter=chapter_plan,
                    lesson_plan=lesson_plan,
                    error=error,
                )
                messages = Conversation.from_message_list(
                    [AI_TUTOR_MSG, Message(role="user", content=prompt)]
                )
                q_plan = LLM("@planQuestions").chat_completion(messages=messages)

                # Convert questions to objects:
                prompt = self.prompts.question_to_obj_prompt.replace(
                    question_data=q_plan,
                )
                messages = Conversation.from_message_list(
                    [AI_TUTOR_MSG, Message(role="user", content=prompt)]
                )
                q_objs = LLM("@objConvertQuestions").chat_completion(messages=messages)

                # Try to create questions from the question objects
                try:
                    q_objs = JSONExtractor.extract(q_objs)
                except Exception as e:
                    error = f"Error while creating Question Objects: {e}"
                    continue

                # Check if the output is a list of questions
                if isinstance(q_objs, list) and all(
                    isinstance(q, dict) for q in q_objs
                ):
                    # Check to make sure that each question object is valid
                    for q in q_objs:
                        subjects = [t for t in list(Question.Subject) if t.name.lower() == q.get("subject").lower()]
                        types = [t for t in list(Question.Type) if t.name.lower() == q.get("type").lower()]
                        assert len(subjects) == 1, f"Error, could not find subject on question, check the input: {str(q)}"
                        assert len(types) == 1, f"Error, could not find type on question, check the input: {str(q)}"
                        assert q.get(
                            "concepts"
                        ), f"Error, could not find Concept Database Mappings on Question JSON object. check the input: {str(q)}"
                        assert q.get(
                            "plan"
                        ), f"Error, could not find Plan on Question JSON object. check the input: {str(q)}"
                        q_obj = Question(
                            q_subject=subjects[0],
                            q_type=types[0],
                            question_data={"plan": q.get("plan")},
                            concepts=[
                                self.ConceptDatabase.get_concept(c)
                                for c in q.get("concepts")
                            ],
                        )
                        self.Questions.append(q_obj)

                # Save the question plan
                output_dir = "training_data/questions/plan"
                save_training_data(
                    output_dir, json_to_string(messages.format_json()), q_plan
                )

                self.num_slides = len(q_objs)
                return
            except Exception as e:
                error = f"Error while creating a Question Plan: {e}"

    def generate_question_suite(self, chapter_plan: str, lesson_plan: str):
        self.generate_questions(chapter_plan, lesson_plan)
        # Create the questions from the question objects in a threaded manner
        threads = []
        generation_lock = threading.Lock()
        for q_obj in self.Questions:
            thread = threading.Thread(
                target=self.generate_question_data,
                args=(q_obj, generation_lock),
            )
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

    def env_string(self):
        return (
            "\n\n".join(
                [
                    f"**{str(i)}**:\n" + question.env_string()
                    for i, question in enumerate(self.Questions)
                ]
            )
            if self.Questions
            else "There are no Questions created currently."
        )

    def to_sql(self):
        return (
            self.current_obj_idx,
            self.num_questions,
            [question.format_json() for question in self.Questions],
        )

    def combine_prompts(self, subject, q_type):
        main_prompt = self.prompts.main_prompt.get_prompt()
        subject_instructions = self.prompts.subject_prompts[subject]
        type_instructions = self.prompts.type_prompts[q_type]

        return main_prompt.replace(
            "$SUBJECT_INSTRUCTIONS$", subject_instructions
        ).replace("$TYPE_INSTRUCTIONS$", type_instructions)

    @staticmethod
    def from_sql(
        current_obj_idx,
        num_questions,
        questions: List[Tuple[int, int, str, List[str]]],
        Notebank,
        ConceptDatabase,
    ):
        q_suite = QuestionSuite(num_questions, Notebank, ConceptDatabase)
        q_suite.current_obj_idx = current_obj_idx
        q_suite.Questions = [
            Question.from_sql(
                q[0],
                q[1],
                q[2],
                q[3],
                [ConceptDatabase.get_concept(cpt) for cpt in q[4]],
            )
            for q in questions
        ]
        return q_suite

    def get_object(self, idx):
        assert (
            0 <= idx < self.num_questions
        ), "Cannot access Question Object Array Out of Bounds"
        return self.Questions[idx]

    def format_json(self):
        return {
            "questions": [question.format_json() for question in self.Questions],
            "current_obj_idx": self.current_obj_idx,
            "num_questions": self.num_questions,
        }

    def is_completed(self):
        return all([question.completed for question in self.Questions])


class Question(JSONSerializable, SQLSerializable):
    class Subject(IntEnum):
        MATH = 0
        CODE = 1
        LITERATURE = 2
        CONCEPTUAL = 3

    class Type(IntEnum):
        TEXT_ENTRY = 0
        MULTIPLE_CHOICE = 1
        CALCULATION_ENTRY = 2
        CODE_ENTRY = 3

    MAP_SUBJECT_2_STR = {
        0: "Math (0)",
        1: "Code (1)",
        2: "Literature (2)",
        3: "Conceptual (3)",
    }
    MAP_TYPE_2_STR = {
        0: "TEXT_ENTRY (0)",
        1: "MULTIPLE_CHOICE (1)",
        2: "CALCULATION_ENTRY (2)",
        3: "CODE_ENTRY (3)",
    }

    def __init__(
        self,
        q_subject: "Question.Subject",
        q_type: "Question.Type",
        question_data: dict,
        concepts: List["Concept"],
        student_response: dict = {},
        completed: bool = False,
    ):
        self.subject: Question.Subject = q_subject
        self.type: Question.Type = q_type
        self.data: dict = question_data
        self.concepts = [c for c in concepts if c is not None]

        self.student_response = student_response
        self.completed = completed

    def __repr__(self) -> str:
        return f"Question(data: {self.data})"

    def format_json(self):
        return {
            "subject": self.subject.name,
            "type": self.type.name,
            "data": self.data.copy(),
            "concepts": [c.name for c in self.concepts],
            "student_response": self.student_response,
            "completed": self.completed,
        }
    
    def env_string(self):
        concepts_list = "\n - ".join([f"`{concept.name}`" for concept in self.concepts])
        return (f"### Question:\n" +
                f" **Subject**: {self.subject.name}\n" +
                f" **Type**: {self.type.name}\n" +
                f" **Concepts**: \n" +
                f"    - {concepts_list}"
                )

    def evaluate(self, student_response_data: dict) -> dict:
        self.student_response = student_response_data
        return_data = {}

        if self.type == Question.Type.TEXT_ENTRY:
            self.completed = True  # TODO: Implement Text Entry Evaluation
        elif self.type == Question.Type.MULTIPLE_CHOICE:
            self.completed = self.data.get(
                "correct_entry"
            ) == self.student_response.get(
                "entry"
            )  # TODO: Implement Multiple Choice Evaluation
            return_data["feedback"] = (
                "Correct!" if self.completed else "incorrect..."
            )  # TODO: Implement Multiple Choice Feedback by using Chat messaging
        elif self.type == Question.Type.CALCULATION_ENTRY:
            self.completed = True  # TODO: Implement Calculation Entry Evaluation
        elif self.type == Question.Type.CODE_ENTRY:
            self.completed = True  # TODO: Implement Code Entry Evaluation

        return_data["completed"] = self.completed

        return return_data
