import json
import os
import re
from enum import IntEnum
from typing import List

from AITutor_Backend.src.BackendUtils.json_serialize import *
from AITutor_Backend.src.BackendUtils.llm_client import LLM

from AITutor_Backend.src.PromptUtils.prompt_template import PromptTemplate
from AITutor_Backend.src.PromptUtils.prompt_utils import (
    Message,
    Conversation,
    AI_TUTOR_MSG,
)

from AITutor_Backend.src.DataUtils.file_utils import save_training_data

from AITutor_Backend.src.TutorUtils.chat_history import ChatHistory
from AITutor_Backend.src.TutorUtils.notebank import NoteBank

DEBUG = bool(os.environ.get("DEBUG", 0))


class Prompter:
    CURR_ENV_NOTEBANK_DELIMITER = "$NOTEBANK.STATE$"  # Environment for the notebankd
    CURR_ENV_CHAT_HISTORY_DELIMITER = (
        "$CHAT_HISTORY$"  # Environment for the chat history
    )
    QUESTION_COUNTER_DELIMITER = "$NUM_QUESTIONS$"
    PLAN_DELIMITER = "$ACTION_PLAN$"
    CURR_ERROR_DELIMITER = "$CURR_ENV.ERROR$"

    def __init__(
        self,
        question_prompt_file,
        notebank_prompt_file,
        plan_prompt_file,
        notebank: NoteBank,
        chat_history: ChatHistory,
        debug=False,
    ):
        self.notebank = notebank
        self.chat_history = chat_history
        self.__questions_asked = 0

        # Question Creation Prompt:
        self._question_prompt_template = PromptTemplate.from_file(
            question_prompt_file,
            {
                "env_notebank": Prompter.CURR_ENV_NOTEBANK_DELIMITER,
                "chat_history": Prompter.CURR_ENV_CHAT_HISTORY_DELIMITER,
                "question_counter": Prompter.QUESTION_COUNTER_DELIMITER,
                "curr_error": Prompter.CURR_ERROR_DELIMITER,
                "plan": Prompter.PLAN_DELIMITER,
            },
        )
        """vars: env_notebank, chat_history, question_counter, curr_error, plan"""

        # Registers Notebank Action Prompt:
        self._notebank_prompt_template = PromptTemplate.from_file(
            notebank_prompt_file,
            {
                "env_notebank": Prompter.CURR_ENV_NOTEBANK_DELIMITER,
                "chat_history": Prompter.CURR_ENV_CHAT_HISTORY_DELIMITER,
                "question_counter": Prompter.QUESTION_COUNTER_DELIMITER,
                "curr_error": Prompter.CURR_ERROR_DELIMITER,
                "plan": Prompter.PLAN_DELIMITER,
            },
        )
        """vars: env_notebank, chat_history, question_counter, curr_Error, plan"""

        # Registers Plan Prompt:
        self._plan_prompt_template = PromptTemplate.from_file(
            plan_prompt_file,
            {
                "env_notebank": Prompter.CURR_ENV_NOTEBANK_DELIMITER,
                "env_chat_history": Prompter.CURR_ENV_CHAT_HISTORY_DELIMITER,
                "question_counter": Prompter.QUESTION_COUNTER_DELIMITER,
            },
        )
        """vars: env_notebank, env_chat_history and question_counter"""

    def perform_plan(
        self,
    ):
        """
        Get Plan from a LLM
        """
        # load prompt data
        prompt = self._plan_prompt_template.replace(
            env_notebank=self.notebank.env_string(),
            env_chat_history=self.chat_history.env_string(),
            question_counter=str(self.__questions_asked),
        )

        # Create conversation with AI Tutor Sys Prompt.
        messages = Conversation.from_message_list(
            [AI_TUTOR_MSG, Message("user", prompt)]
        )

        llm_plan = LLM("claude-3-opus-20240229").chat_completion(messages)

        output_dir = "training_data/prompter/planning/"
        save_training_data(output_dir, prompt, llm_plan)
        return llm_plan

    def perform_notebank(self, plan):
        """
        Get Notebank Action from a LLM
        """
        error = "There is no current error."
        while True:
            # with open("translation.txt", "a") as f:
            # f.write("TRANSLATION\n")
            # Create prompt
            prompt = self._notebank_prompt_template.replace(
                env_notebank=self.notebank.env_string(),
                chat_history=self.chat_history.env_string(),
                question_counter=str(self.__questions_asked),
                curr_error=error,
                plan=plan,
            )

            # Create conversation with AI Tutor Sys Prompt.
            messages = Conversation.from_message_list(
                [AI_TUTOR_MSG, Message("user", prompt)]
            )

            llm_output = LLM("gpt-3.5-turbo-16k").chat_completion(messages)
            # TODO: Continue from here editing this file to update

            success, error, terminate = self.notebank.process_llm_action(llm_output)
            # with open("translation.txt", "a") as f:
            #     f.write("TRANSLATION\n")

            if success or terminate:
                output_dir = "training_data/prompter/notebank/"
                save_training_data(output_dir, prompt, llm_output)
                break

            # with open("translation_errors.txt", "a") as f:
            #     f.write("TRANSLATION_ERROR\n") # Used in error calculation
        return terminate

    def get_prompting(self, plan):
        """
        Get Prompt Question from a LLM
        """
        error = "There is no current error."
        while True:
            try:
                with open("translation.txt", "a") as f:
                    f.write("TRANSLATION\n")

                # Load prompt
                prompt = self._question_prompt_template.replace(
                    env_notebank=self.notebank.env_string(),
                    chat_history=self.chat_history.env_string(),
                    question_counter=str(self.__questions_asked),
                    curr_error=error,
                    plan=plan,
                )

                # Create conversation with AI Tutor Sys Prompt.
                messages = Conversation.from_message_list(
                    [AI_TUTOR_MSG, Message("user", prompt)]
                )

                llm_output = LLM("gpt-3.5-turbo-16k").chat_completion(messages)

                action = PromptAction.parse_llm_action(llm_output)

                output_dir = "training_data/prompter/prompt/"
                save_training_data(output_dir, prompt, llm_output)
                break
            except Exception as e:
                error = "There was an error while trying to parse the Prompt: " + str(e)
                # with open("translation_errors.txt", "a") as f:
                #     f.write("TRANSLATION_ERROR\n")

        # Return the PromptAction parsed from the LLM:
        self.__questions_asked += 1
        return action

    def perform_tutor(self, student_input: str):
        """
        Cognitive Process
        """
        # Add msg to chat:
        self.chat_history.hear(student_input)

        # Construct the notebank:
        plan = self.perform_plan()
        if DEBUG:
            print(f"\n[Plan]\n{plan}\n[/PLAN]\n")
        terminate = self.perform_notebank(plan)

        # Construct the prompting:
        llm_prompt = None
        if not terminate:
            llm_prompt = self.get_prompting(plan)
            self.chat_history.respond(llm_prompt._data)
            terminate = llm_prompt._type == PromptAction.Type.TERMINATE
        return llm_prompt, terminate


class PromptAction(
    JSONSerializable,
):
    class Type(IntEnum):
        FILE = 0
        TEXT = 1
        RATING = 2
        TERMINATE = -1

    __QUESTION_REGEX = re.compile(r"\`\`\`json([^\`]*)\`\`\`")

    def __init__(
        self, prompt: str, type: "PromptAction.Type", suggested_responses: List[str]
    ):
        self._type = type
        self._data = prompt
        self._suggested_responses = suggested_responses

    def format_json(self):
        """
        Format into JSON Object:
        - type: ENUM (0=FILE, 1=TEXT, 2=RATING, -1=TERMINATE)
        - question: str
        """
        return {
            "type": int(self._type),
            "question": self._data,
            "suggested_responses": self._suggested_responses.copy(),
        }

    @staticmethod
    def parse_llm_action(llm_output: str) -> "PromptAction":
        """
        Given LLM Output; parse for formattable Prompt Type and Question
        """
        # Search for action in llm output
        regex_match = PromptAction.__QUESTION_REGEX.findall(llm_output)
        # Try to get json format or attempt to use output as json
        if regex_match:
            regex_match = (
                regex_match[0].replace("```json", "").replace("```", "").strip()
            )

        # Try to extract and load JSON data from the
        prompt_data = regex_match if regex_match else llm_output
        try:
            action_data = json.loads(prompt_data)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Error parsing JSON on output: {llm_output},  error: {str(e)}"
            )

        action_type = action_data.get("type").lower()
        prompt = action_data.get("prompt")
        s_responses = action_data.get("suggested_responses", [])

        # Map the action type to the corresponding Type enum, validate the type and prompt data
        type_map = {
            "file": PromptAction.Type.FILE,
            "text": PromptAction.Type.TEXT,
            "rating": PromptAction.Type.RATING,
            "terminate": PromptAction.Type.TERMINATE,
        }

        p_type = type_map.get(action_type, None)
        assert p_type is not None, "Error: Unknown action type."
        assert prompt, "Error: Prompt text is missing."

        # Create action and assert data creation is valid
        action = PromptAction(prompt, p_type, s_responses)

        assert isinstance(
            action._type, PromptAction.Type
        ), "Error while Creating the Prompt."
        assert action._data, "Error while parsing the data for the Prompt."

        return action
