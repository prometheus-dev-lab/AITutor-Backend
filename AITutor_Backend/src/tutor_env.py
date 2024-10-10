import os
import pickle as pkl
import threading
from enum import IntEnum
from typing import Dict

from AITutor_Backend.src.TutorUtils.chat_history import *
from AITutor_Backend.src.TutorUtils.concepts import ConceptDatabase
from AITutor_Backend.src.TutorUtils.tutor_objs import Chapter, Lesson, TutorObjManager
from AITutor_Backend.src.TutorUtils.Modules.questions import QuestionSuite
from AITutor_Backend.src.TutorUtils.Modules.slides import SlidePlanner
from AITutor_Backend.src.TutorUtils.notebank import *
from AITutor_Backend.src.TutorUtils.prompts import PromptAction, Prompter


from AITutor_Backend.src.BackendUtils.extractors import JSONExtractor
from AITutor_Backend.src.BackendUtils.llm_client import LLM

from AITutor_Backend.src.PromptUtils.prompt_template import PromptTemplate
from AITutor_Backend.src.PromptUtils.prompt_utils import (
    Message,
    Conversation,
    AI_TUTOR_MSG,
)


DEBUG_GENERATION_DATA = bool(os.environ.get("DEBUG_FRONTEND", False))
SESSIONS_DATA_PATH = "AITutor_Backend/data/saved_sessions/Agent AI"


class TutorEnv(  ################# TUTOR_ENV #######################
    SQLSerializable,
):
    class States(IntEnum):
        PROMPTING = 0
        TEACHING = 1
        TESTING = 2
        METRICS = 3  # TODO: IMPLEMENT
        GENERATION = 4

    class Prompter(  ################# PROMPTER #######################
        SQLSerializable,
    ):
        def initial_generation_pipeline(self):
            """Initially generated the resources needed by the Tutor. Initializes objects."""
            pass

        def __init__(
            self,
            env: "TutorEnv",
            main_concept_file,
            concept_list_file,
            notebank_filter_file,
        ):
            super(TutorEnv.Prompter, self).__init__()
            self.env = env

            # Load in Prompt file data:
            with open(main_concept_file, "r") as f:
                self.__main_concept_prompt = f.read()
            with open(concept_list_file, "r") as f:
                self.__concept_list_prompt = f.read()
            with open(notebank_filter_file, "r") as f:
                self.__notebank_filter_prompt = f.read()

        def __get_main_concept(
            self,
        ):
            """Uses the notebank to extract the main concept"""
            # Load Data:
            messages = Conversation.from_message_list(
                [
                    Message(role="system", content=self.__main_concept_prompt),
                    Message(
                        role="user",
                        content=f"// Input:\n {self.env.notebank.env_string()}\n\n// Output:",
                    ),
                ]
            )

            # Get validated response:
            while True:
                llm_output = LLM("gpt-3.5-turbo-16k").chat_completion(
                    messages=messages,
                    max_tokens=256,
                )

                try:
                    main_concept = JSONExtractor.extract(llm_output)["main_concept"]
                    return main_concept

                except:  # Just retry on fail
                    pass

        def __get_concept_list(
            self,
        ):
            """Uses the notebank to extract the concept list"""
            # Load Data:
            messages = Conversation.from_message_list(
                [
                    Message(
                        role="system",
                        content=self.__concept_list_prompt,
                    ),
                    Message(
                        role="user",
                        content=f"// Input:\n {self.env.notebank.env_string()}\n\n// Output:",
                    ),
                ]
            )

            # Get validated response:
            while True:
                llm_output = LLM("gpt-3.5-turbo-16k").chat_completion(
                    messages=messages,
                    max_tokens=8000,
                )

                try:
                    concept_list = JSONExtractor.extract(llm_output)["concept_list"]
                    return concept_list

                except:  # Just retry
                    pass

        def __get_filtered_notebank(
            self,
        ):
            """Uses the notebank to filter out all of the entries"""
            # Load Data:
            messages = Conversation.from_message_list(
                [
                    Message(
                        role="system",
                        content=self.__notebank_filter_prompt,
                    ),
                    Message(
                        role="user",
                        content=f"// Input:\n {self.env.notebank.env_string()}\n\n// Output:",
                    ),
                ]
            )

            # Get validated response:
            while True:
                llm_output = LLM("gpt-3.5-turbo-16k").chat_completion(
                    messages=messages,
                    max_tokens=8000,
                )

                try:
                    notes = llm_output.split("\n")
                    if len(notes) > 3:
                        return notes

                except:  # Just retry
                    pass

        def process_action(self, user_input):
            """Processes user input and returns new state and Tutor Actions

            Args:
                user_input (Dict[str, any]): JSON data from user

            Returns:
                new_state: data to return to the user
                current_env_state: enum value which iterates through the environment states
            """
            if user_input.get("is_audio", False):  # User Data is Audio
                pass
                # TODO: user_input["user_prompt"] = self.__ears.read_chat(user_input["user_prompt"])

            user_prompt = user_input["user_prompt"]
            # TODO: Do processing

            ### PROMPTING PHASE
            if self.env.current_state == int(TutorEnv.States.PROMPTING):
                # Get tutor update:
                prompt_obj, terminate = self.env.prompter.perform_tutor(user_prompt)

                # check if the intent is to continue with the next step:
                if terminate or prompt_obj._type == PromptAction.Type.TERMINATE:
                    print(self.env.notebank.env_string())
                    concept_list = self.__get_concept_list()
                    print(concept_list)

                    # TODO: Implement generation
                    prompt_obj = PromptAction(
                        "[SEP]".join(concept_list), PromptAction.Type.TERMINATE, []
                    )  # fix to return teaching objects

                    self.env.current_state = int(TutorEnv.States.GENERATION)

                return prompt_obj.format_json()
            ### END PROMPTING PHASE

            ### GENERATION PHASE
            if self.env.current_state == TutorEnv.States.GENERATION:
                # Add new concepts:
                if "list_concepts" in user_input:
                    for concept in user_input["list_concepts"]:
                        self.env.notebank.add_note(f"Concept: {concept}")
                else:
                    Warning

                if "student_interests" in user_input:
                    self.env.notebank.add_note(
                        f"Student's Interest Statement: {user_input['student_interests']}"
                    )

                if "student_slides" in user_input:
                    self.env.notebank.add_note(
                        f"Student's Slides Preference Statement: {user_input['student_slides']}"
                    )

                if "student_questions" in user_input:
                    self.env.notebank.add_note(
                        f"Student's Questions Preference Statement: {user_input['student_questions']}"
                    )

                main_concept = self.__get_main_concept()
                self.env.notebank.add_note(f"Main Concept: {main_concept}")

                # Filter Notebank:
                notes = self.__get_filtered_notebank()
                self.env.notebank.clear()

                # iterate through notes and add to Notebank
                for note in notes:
                    self.env.notebank.add_note(note)

                # Generate Concept Database:
                self.env.concept_database = ConceptDatabase(
                    main_concept,
                    self.env.notebank.env_string(),
                )

                if not DEBUG_GENERATION_DATA:
                    # Generate Concept Database:
                    self.env.concept_database.generate_concept_graph()
                    self.env.concept_database.generate_concepts()

                    self.env.obj_manager = TutorObjManager(
                        self.env.notebank, self.env.concept_database
                    )

                    self.env.obj_manager.generate_chapters()  # Generates C Chapters
                    self.env.obj_manager.initialize_chapter(
                        0
                    )  # Takes Chapter[0], Creates L Lessons => (Outcomes, Slides, Questions)

                else:  # This is used for Debugging the Frontend Application: set Environment Variable DEBUG_FRONTEND=1
                    self.populate_data_from_file(self.env)

                # obj_idx = self.env.slide_planner.current_obj_idx
                # TODO: Finish implementing object control in backend
                # current_chapter
                #   current_lesson
                #       current_chapter
                #       current_question

                learning_obj = {
                    "conversational_response": "Great! Now we can start learning. ",
                    "obj_manager": "",
                }

                self.env.chat_history.respond(learning_obj["conversational_response"])

                self.env.current_state = TutorEnv.States.TEACHING

                return learning_obj
            ### END GENERATION PHASE

            # self.env.current_state = int(user_input["current_state"])

            ### TEACHING PHASE
            if self.env.current_state == int(TutorEnv.States.TEACHING):

                obj_idx = user_input.get(
                    "obj_idx", self.env.slide_planner.current_obj_idx
                )
                if obj_idx == self.env.slide_planner.current_obj_idx:
                    # TODO: listen to user user_input and create a response
                    self.env.chat_history.hear(user_prompt)
                    # ...
                    learning_obj = self.env.slide_planner.format_json()
                    learning_obj["conversational_response"] = "todo: implement"
                else:
                    self.env.slide_planner.current_obj_idx = obj_idx
                    learning_obj = self.env.slide_planner.format_json()
                    learning_obj["conversational_response"] = (
                        self.env.slide_planner.get_object(obj_idx).presentation
                    )
                    self.env.chat_history.respond(
                        learning_obj["conversational_response"]
                    )
                    return learning_obj

                self.env.chat_history.respond(learning_obj["conversational_response"])
                return learning_obj
            ### END TEACHING PHASE

            ### TESTING PHASE
            if self.env.current_state == int(TutorEnv.States.TESTING):
                obj_idx = user_input.get(
                    "obj_idx", self.env.question_suite.current_obj_idx
                )
                if obj_idx == self.env.question_suite.current_obj_idx:
                    if user_input["submit_response"]:
                        # TODO: Evaluate the response the the question
                        return self.env.question_suite.format_json().update(
                            {"conversational_response": "todo: implement"}
                        )
                    else:
                        # TODO: listen to user user_input and create a response
                        self.env.chat_history.hear(user_prompt)
                    # ...
                    testing_obj = self.env.question_suite.format_json().update(
                        {"conversational_response": "todo: implement"}
                    )
                else:
                    self.env.question_suite.current_obj_idx = obj_idx
                    testing_obj = self.env.question_suite.format_json().update(
                        {
                            "conversational_response": self.env.question_suite.get_object(
                                obj_idx
                            ).presentation
                        }
                    )
                self.env.chat_history.respond(testing_obj["conversational_response"])
                return testing_obj
            ### END TESTING PHASE

    ################# ENDOF PROMPTER #######################

    def __init__(
        self,
    ):
        """Creates base TutorEnv

        TutorEnv has a:
            - prompter
            - notebank
            - chat history
            - concept database
            - Obj Manager

        """
        super(TutorEnv, self).__init__()
        self.current_state = int(TutorEnv.States.PROMPTING)  # Prompt Start
        self.notebank = NoteBank()
        self.chat_history = ChatHistory()
        self.prompter = Prompter(
            "AITutor_Backend/src/TutorUtils/Prompts/PromptingPhase/question_prompt",
            "AITutor_Backend/src/TutorUtils/Prompts/PromptingPhase/notebank_prompt",
            "AITutor_Backend/src/TutorUtils/Prompts/PromptingPhase/prompt_plan_prompt",
            self.notebank,
            self.chat_history,
        )
        self.concept_database = None
        self.obj_manager = None

        self._content_generated = False

        self.prompter = TutorEnv.Prompter(
            self,
            "AITutor_Backend/src/TutorUtils/Prompts/KnowledgePhase/Concepts//main_concept_prompt",
            "AITutor_Backend/src/TutorUtils/Prompts/KnowledgePhase/Concepts/concept_list_prompt",
            "AITutor_Backend/src/TutorUtils/Prompts/KnowledgePhase/notebank_filter_prompt",
        )

    def step(self, input_data: Dict[str, any]):
        return self.prompter.process_action(input_data), self.current_state

    ## Data Functions:
    @staticmethod
    def from_sql(current_state, notebank_state, chat_history):
        """Recreates a Chat History from an SQL value

        Args:
            chat_history (str): \'[SEP]\'  Serialized version of Chat History.
        """
        tutor_ref = TutorEnv()
        tutor_ref.current_state = current_state
        tutor_ref.notebank = NoteBank.from_sql(notebank_state)
        tutor_ref.chat_history = ChatHistory.from_sql(chat_history)
        tutor_ref.prompter.notebank = tutor_ref.notebank
        tutor_ref.prompter.chat_history = tutor_ref.chat_history
        return tutor_ref

    def to_sql(
        self,
    ) -> str:
        """Serializes a Chat History for SQL

        Returns:
            str: Chat History
        """
        return self.current_state

    @classmethod
    def populate_data_from_file(cls) -> "TutorEnv":
        # Generated env data used for testing different components and new features:
        main_concept = "Agent AI"
        notebank_data = """User expresses interest in learning about agent AI.
        Main Concept: Agent AI
        Student wants to learn about agent AI
        Tutor needs to gauge student's background knowledge
        in artificial intelligence and computer science.
        Tutor should ask student about their specific interests in agent AI anany particular agent types or applications they want to learn about.
        Tutor should inquire about the student's goals in learning about agent AI.
        Tutor should ask student about their preference fo
        a theoretical or practical approach to learning agent AI.
        Tutor should ask student about their familiarity with programming languages or tools used in AI development.
        Tutor to ask student about their familiarity with programming languageand tools used in AI development.
        Tutor to ask student about their specific interest
        in agent AI and any particular agent types or applications they want to learn about.
        Tutor to ask student about their goals in learning about agent AI.
        Tutor to ask student about their preference for a theoretical or practical approach to learning agent AI.
        Tutor should gauge student's current understanding of agent AI concept to create a targeted learning plan.
        Tutor should document their responses and preferences in the Notebank for future reference.
        Concept: Introduction to Artificial Intelligence
        Concept: Definition and Characteristics of Agents
        Concept: Agent Architectures
        Concept: Simple Reflex Agents
        Concept: Model-Based Reflex Agents
        Concept: Goal-Based Agents
        Concept: Utility-Based Agents
        Concept: Learning Agents
        Concept: Agent Environments
        Concept: Classifications of Environments
        Concept: Design and Simulation of Environments
        Concept: Agent Communication and Coordination
        Concept: Communication Protocols and Languages
        Concept: Coordination Mechanisms
        Concept: Collaborative and Competitive Agent Interactions
        Concept: Learning Agents and Adaptive Behavior
        Concept: Machine Learning Techniques in Agents
        Concept: Reinforcement Learning
        Concept: Multi-Agent Systems
        Concept: Design and Management of Multi-Agent Systems
        Concept: Cooperation and Competition among Agents
        Concept: Intelligent Agents in Games, Robotics, and Simulation
        Concept: Agents in Video Games
        Concept: Robotic Agents
        Concept: Simulation Models Using Agents
        Concept: Ethical Considerations and Future Trends in Agent AI
        Concept: Ethical Implications of Autonomous Agents
        Concept: Future Developments and Emerging Technologies
        Concept: Student's Background Knowledge
        Concept: Specific Interests in Agent AI
        Concept: Learning Goals
        Concept: Theoretical vs. Practical Approach Preference
        Concept: Familiarity with AI Programming Languages and Tools
        Concept: Targeted Learning Plan Based on Current UnderstandinG
        """

        tutor_env = cls()

        # Load notebank:
        tutor_env.notebank = NoteBank.from_sql(notebank_data)

        # Load Concept Database:
        tutor_env.concept_database = ConceptDatabase(
            main_concept, tutor_env.notebank.env_string()
        )

        if os.path.exists(os.path.join(SESSIONS_DATA_PATH, "concept_database.pkl")):
            with open(
                os.path.join(SESSIONS_DATA_PATH, "concept_database.pkl"), "rb"
            ) as f:
                tutor_env.concept_database = pkl.load(f)

        tutor_env.obj_manager = TutorObjManager(
            tutor_env.notebank, tutor_env.concept_database
        )

        # Load Object Manager:
        if os.path.exists(os.path.join(SESSIONS_DATA_PATH, "obj_manager.pkl")):
            with open(os.path.join(SESSIONS_DATA_PATH, "obj_manager.pkl"), "rb") as f:
                tutor_env.obj_manager = pkl.load(f)

        if tutor_env.obj_manager.num_chapters > 0:
            tutor_env.obj_manager.curr_chapter_idx = 0

        if tutor_env.obj_manager.Chapters[0].num_lessons > 0:
            tutor_env.obj_manager.curr_lesson_idx = 0

        return tutor_env

    ################# ENDOF TUTOR_ENV #######################
