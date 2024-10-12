import copy
from AITutor_Backend.src.BackendUtils.json_serialize import JSONSerializable
from dataclasses import dataclass
from typing import List


@dataclass
class Message(JSONSerializable):
    role: str
    """the role of the message sender"""

    content: str
    """message's data"""

    def to_dict(self):
        return {"role": self.role, "content": self.content}

    def format_json(self):
        return self.to_dict()

    def copy(self) -> "Message":
        """Returns a copied Message"""
        n_message = Message(None, None)
        n_message.__dict__ = copy.deepcopy(self.__dict__)

        return n_message


@dataclass
class Conversation(JSONSerializable):
    def __post_init__(self):
        self.messages = []

    def append_message(self, message: Message):
        self.messages.append(message)

    @classmethod
    def from_message_list(cls, messages: List[Message]) -> "Conversation":
        """Copies data from messages into new Conversation instance"""
        convo = cls()
        convo.messages = [m.copy() for m in messages]
        return convo

    def to_dict(self):
        return [m.to_dict() for m in self.messages]

    def format_json(self):
        return {"messages": self.to_dict()}

    def copy(self) -> "Conversation":
        """Returns a copied Message"""
        n_conversation = Conversation()
        n_conversation.messages = [m.copy() for m in self.messages]

        return n_conversation


AI_TUTOR_MSG = Message("system", "Act as an intelligent AI Tutor.")
