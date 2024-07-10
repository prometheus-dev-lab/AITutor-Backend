from AITutor_Backend.src.PromptUtils.prompt_utils import Conversation, Message
from litellm import completion
from typing import Optional
import os


class LLM:
    def __init__(
        self,
        model_id: str,
        base_url: str = None,
    ):
        self._model_id = model_id
        self._base_url = base_url

    def chat_completion(
        self, messages: Conversation, max_tokens: Optional[int] = None
    ) -> str:
        """
        Recieves a Conversation and returns a message from the assistant
        """
        response = completion(
            self._model_id,
            messages.to_dict(),
            base_url=self._base_url,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content
