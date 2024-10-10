import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from AITutor_Backend.src.BackendUtils.json_serialize import JSONSerializable
from AITutor_Backend.src.BackendUtils.env_serialize import EnvSerializable
from AITutor_Backend.src.BackendUtils.sql_serialize import SQLSerializable
from AITutor_Backend.src.BackendUtils.llm_client import LLM
from AITutor_Backend.src.PromptUtils.prompt_template import PromptTemplate
from AITutor_Backend.src.PromptUtils.prompt_utils import (
    Message,
    Conversation,
    AI_TUTOR_MSG,
)
import numpy as np
from AITutor_Backend.src.models.language_models import Ranker

from math import log


@dataclass
class Note(EnvSerializable):
    emitter: str
    data: str

    def env_string(self):
        return "### Note:\n" + f" - **Created by:** {self.emitter}\n" + f'"{self.data}"'


class NoteBank(JSONSerializable, EnvSerializable, SQLSerializable):
    def __init__(self):
        self.__notes: List[Note] = []
        self.__cache_size = 1000
        self.__ranker = Ranker(cache_size=self.__cache_size)
        # self.__prune_prompt = PromptTemplate.from_config("@pruneNotes", {
        #     "context": "$CONTEXT$",
        #     "objective": "$OBJECTIVE$",
        # })
        self._note_prompt = PromptTemplate.from_config(
            "@noteGeneration",
            {
                "context": "$CONTEXT$",
                "objective": "$OBJECTIVE$",
            },
        )

    def add_note(self, note: Note):
        """Adds a note to the NoteBank"""
        self.__notes.append(note)

    def generate_note(self, emitter: str, context: str, objective: str) -> None:
        """Generates a note based on the given context and objective"""
        prompt = self._note_prompt.replace(context=context, objective=objective)
        messages = Conversation.from_message_list(
            [AI_TUTOR_MSG, Message(role="user", content=prompt)]
        )
        note_summary = LLM("@noteGeneration").chat_completion(messages=messages)
        note = Note(emitter, note_summary)
        self.add_note(note)

    def query_context_and_generate_summary(
        self, emitter: str, context: str, objective: str
    ) -> None:
        """Generates a note based on the given context and objective"""
        prompt = self._note_prompt.replace(context=context, objective=objective)
        messages = Conversation.from_message_list(
            [AI_TUTOR_MSG, Message(role="user", content=prompt)]
        )
        note_summary = LLM("@noteGeneration").chat_completion(messages=messages)
        note = Note(emitter, note_summary)
        self.add_note(note)

    def add_note(self, emitter: str, data: str) -> None:
        """Adds a note to the NoteBank"""
        note = Note(emitter, data)
        self.__notes.append(note)

        # Prune the notes if the cache size is too large, we use a logarithmic factor to avoid pruning too often and collecting too many notes
        if len(self.__notes) > (self.__cache_size + log(base=2, x=self.__cache_size)):
            self.prune_notes()

    def prune_notes(self) -> None:
        """Prune the notes to the cache size"""
        pass  # TODO: Implement the functionality of pruning the notes

    def generate_context_summary(
        self, query: str, objective: Optional[str], k: int = 10
    ) -> str:
        """Generates a summary of the context"""
        objective = (
            objective
            or f"Generate a summary of the context based on the query: {query}"
        )
        context = self.get_top_k_results(query, k=k)
        prompt = self._note_prompt.replace(context=context, objective=objective)
        messages = Conversation.from_message_list(
            [AI_TUTOR_MSG, Message(role="user", content=prompt)]
        )
        note_summary = LLM("@noteGeneration").chat_completion(messages=messages)
        return note_summary

    def get_top_k_results(
        self, query: str, context: str = "N/A", k: int = 10
    ) -> List[Note]:
        """Returns the top k notes ranked based on similarity to the query"""
        if not self.__notes:
            return []

        texts = [note.env_string() for note in self.__notes]
        scores = self.__ranker.rank(query, texts)

        # Combine notes with their scores and sort
        ranked_notes = sorted(
            zip(self.__notes, scores), key=lambda x: x[1], reverse=True
        )

        # Return top k notes
        return [note for note, _ in ranked_notes[:k]]

    def clear(self):
        """Clears the NoteBank's content."""
        self.__notes = []

    def size(self) -> int:
        """Returns size of the NoteBank"""
        return len(self.__notes)

    def get_notes(self) -> List[Note]:
        return self.__notes.copy()

    def format_json(self) -> str:
        return json.dumps(
            {
                "NoteBank": [
                    {"emitter": note.emitter, "data": note.data}
                    for note in self.__notes
                ]
            }
        )

    def to_sql(self) -> str:
        return json.dumps(
            [{"emitter": note.emitter, "data": note.data} for note in self.__notes]
        )

    @staticmethod
    def from_sql(sql_data: str) -> "NoteBank":
        nb = NoteBank()
        notes_data = json.loads(sql_data)
        for note_data in notes_data:
            nb.add_note(Note(emitter=note_data["emitter"], data=note_data["data"]))
        return nb

    def env_string(self) -> str:
        """Returns Environment Observation String which the model will use for prediction."""
        if not self.__notes:
            return "**NoteBank**: The NoteBank is Empty."

        return "**NoteBank**:\n" + "\n".join(
            [
                f"\t{i}. [{note.emitter}] {note.data}"
                for i, note in enumerate(self.__notes)
            ]
        )
