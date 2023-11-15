import re
import openai
from enum import IntEnum
from AITutor_Backend.src.TutorUtils.notebank import NoteBank 
from AITutor_Backend.src.TutorUtils.chat_history import ChatHistory
from AITutor_Backend.src.BackendUtils.json_serialize import *

class Prompter:
    class PrompterLLMAPI:
        CURR_ENV_NOTEBANK_DELIMITER = "$NOTEBANK.STATE$" #Environment for the notebankd
        CURR_ENV_CHAT_HISTORY_DELIMITER = "$CHAT_HISTORY$" #Environment for the chat history 
        
        CURR_ERROR_DELIMITER = "$CURR_ENV.ERROR$"

        def __init__(self, ):
            self.client = openai.OpenAI()
        
        def request_output_from_llm(self, prompt, model: str):
            """Requests the Concept information from an LLM.

            Args:
                prompt: (str) - string to get passed to the model
                model: (str) - 

            Returns:
                _type_: _description_
            """
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    },
                    {
                    "role": "user",
                    "content": "Please carry out whatever task the system is asking you to do, as the AI Tutor our student's education relies it."
                    }
                ],
                temperature=1,
                max_tokens=3000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                
            )

            return response.choices[0].message.content

        def __load_prompt(self, prompt_template, state_dict):
            prompt_string = prompt_template
            # Replace Values in Prompt:
            for k, v in state_dict.items():
                prompt_string = self.prompt_string.replace(k, v)

            # Return Prompt:
            return prompt_string


    def __init__(self, question_prompt_file, notebank_prompt_file, notebank:NoteBank, chat_history:ChatHistory, debug=False):
        self.llm_api = self.PrompterLLMAPI() 
        self.notebank = notebank
        self.chat_history = chat_history
        with open(question_prompt_file, "r", encoding="utf-8") as f:
            self.__question_prompt_template = "\n".join(f.readlines())

        with open(notebank_prompt_file, "r", encoding="utf-8") as f:
            self.__notebank_prompt_template = "\n".join(f.readlines())
        
    def perform_notebank(self, ):
        """
        Get Notebank Action from a LLM
        """
        error = "There is no current error."
        while True:
            prompt = self.__load_prompt(self.__notebank_prompt_template, {Prompter.PrompterLLMAPI.CURR_ENV_NOTEBANK_DELIMITER: self.notebank.env_string(), Prompter.PrompterLLMAPI.CURR_ERROR_DELIMITER: error},) 
            llm_output = self.llm_api.request_output_from_llm(prompt, "gpt-3.5-turbo-1106")
            success, error = self.notebank.process_llm_output(llm_output)
            if success: break
        return True

    def get_prompting(self, ):
        """
        Get Prompt Question from a LLM
        """  
        error = "There is no current error."      
        while True:
            try:
                prompt = self.__load_prompt(self.__question_prompt_template, {Prompter.PrompterLLMAPI.CURR_ENV_NOTEBANK_DELIMITER: self.notebank.env_string(), Prompter.PrompterLLMAPI.CURR_ERROR_DELIMITER: error})
                llm_output = self.llm_api.request_output_from_llm(prompt, "gpt-3.5-turbo-1106")
                action = PromptAction.parse_llm_action(llm_output)
                assert isinstance(action._type, PromptAction.Type), "Error while Creating the Prompt."
                assert action._data, "Error while parsing the data for the Prompt."
            except Exception as e:
                error = "There was an error while trying to parse the Prompt: " + str(e)
        # Return the PromptAction parsed from the LLM:
        return action
    
    
    def perform_tutor(self, student_input:str):
        self.chat_history.hear(student_input) # DEBUGONLY: Remove this to include in TutorEnv
        # Construct the notebank:
        llm_notebank = self.get_notebank()
        # Construct the prompting:
        llm_prompt = self.get_prompting()
        return llm_prompt


class PromptAction(JSONSerializable,):
    class Type(IntEnum):
        FILE=0
        TEXT=1
        RATING=2
        TERMINATE=-1
    __QUESTION_REGEX = re.compile(r'\`\`\`([Tt]ext|[Ff]ile|[Rr]ating)[Pp]rompt([^\`]*)\`\`\`|(\[TERM\])') # Matches any Prompt String
    def __init__(self, question:str, type):
        super(PromptAction, self).__init__()
        self._type = type
        self._data = question

    def format_json(self):
        """
        Format into JSON Object:
        - type: ENUM (0-FILE, 1-TEXT, 2-RATING, (NEGATIVE)1-TERMINATE)
        - question: STR
        """
        return {"type":int(self._type), "question":self._data}

    @staticmethod
    def parse_llm_action(llm_output:str,) -> 'PromptAction':
        """Given LLM Output; parse for formattable Prompt Type and Question"""
        regex_match = PromptAction.__QUESTION_REGEX.findall(llm_output)
        assert regex_match, f"Error parsing LLM Output for Prompt:\n {llm_output}"
        # Extract the Prompt Type & Data from the LLM Ouput:
        prompt_data = regex_match[0]
        # Detect if Termination Case:
        if len(prompt_data) == 3 and "[TERM]" in prompt_data[2]: return PromptAction("[TERM]", PromptAction.Type.TERMINATE)
        # Handle Prompt Action:
        p_type, prompt, _ = regex_match[0]
        p_type, prompt = p_type.strip().lower(), prompt.strip()
        # Get prompt type
        p_type = {"file": PromptAction.Type.FILE, "rating": PromptAction.Type.RATING, "text": PromptAction.Type.TEXT}.get(p_type, None)
        assert isinstance(p_type, PromptAction.Type), "Error: Could not parse LLM for Prompt Action Type."
        assert prompt, "Error: Could not parse LLM for Prompt Action data."
        return PromptAction(prompt, p_type)

