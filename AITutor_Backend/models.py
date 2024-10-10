# from AITutor_Backend.src.TutorUtils.prompts import Prompter
import uuid
from typing import Tuple

from asgiref.sync import async_to_sync, sync_to_async
from django.db import models
from django.contrib.postgres.fields import JSONField
import uuid

from AITutor_Backend.src.tutor_env import TutorEnv
from AITutor_Backend.src.TutorUtils.concepts import ConceptDatabase
from AITutor_Backend.src.TutorUtils.Modules.questions import QuestionSuite
from AITutor_Backend.src.TutorUtils.Modules.slides import SlidePlanner


class ConceptDatabaseModel(models.Model):
    main_concept = models.CharField(max_length=200)
    concepts_data = JSONField(
        default=dict
    )  # This will store all concept data as a JSON object

    def __str__(self):
        return f"ConceptDatabase for {self.main_concept}"


class ConceptModel(models.Model):
    concept_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    parent = models.CharField(max_length=200, null=True)
    definition = models.TextField()
    latex = models.TextField()
    refs = models.TextField(null=True)
    concept_database = models.ManyToManyField(
        ConceptDatabaseModel, related_name="concepts"
    )


class QuestionSuiteModel(models.Model):
    num_questions = models.IntegerField()
    current_obj_idx = models.IntegerField()


class QuestionModel(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    subject = models.IntegerField()
    type = models.IntegerField()
    instructions = models.TextField()
    data = models.TextField()
    question_suite = models.ManyToManyField(
        QuestionSuiteModel, related_name="questions"
    )
    concepts = models.TextField()

    def __str__(self):
        return f"Question {self.id} of type {self.type} in subject {self.subject}"


class SlidePlannerModel(models.Model):
    num_slides = models.IntegerField()
    current_obj_idx = models.IntegerField()
    slides_data = JSONField(
        default=list
    )  # This will store all slide data as a JSON array

    def __str__(self):
        return f"SlidePlanner with {self.num_slides} slides"


class NotebankModel(models.Model):
    notes = models.TextField()


class ChatHistoryModel(models.Model):
    chat = models.TextField()


class TutorEnvModel(models.Model):
    notebank = models.OneToOneField("NotebankModel", on_delete=models.CASCADE)
    chat_history = models.OneToOneField("ChatHistoryModel", on_delete=models.CASCADE)
    concept_database = models.ForeignKey(
        "ConceptDatabaseModel", on_delete=models.SET_NULL, null=True, blank=True
    )
    question_suite = models.OneToOneField(
        "QuestionSuiteModel", on_delete=models.SET_NULL, null=True, blank=True
    )
    slide_planner = models.OneToOneField(
        "SlidePlannerModel", on_delete=models.SET_NULL, null=True, blank=True
    )
    curr_state = models.SmallIntegerField()


class SessionModel(models.Model):
    session_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tutor = models.ForeignKey(
        "TutorEnvModel", on_delete=models.CASCADE, related_name="sessions"
    )


class DatabaseManager:
    def __init__(self, session_id):
        self.__session_id = session_id
        # Load the session with the given session ID
        self.session = SessionModel.objects.get(session_id=self.__session_id)
        self.tutor_env_model = self.session.tutor
        self.tutor_env = None
        self.concept_database = None
        self.question_suite = None

    @staticmethod
    def create_tutor_session():
        """Returns tutor_environment_model"""
        notebank = NotebankModel.objects.create(notes="")
        chat_history = ChatHistoryModel.objects.create(chat="")
        tutor_env_model = TutorEnvModel.objects.create(
            notebank=notebank,
            chat_history=chat_history,
            curr_state=0,
        )
        session = SessionModel.objects.create(tutor=tutor_env_model)
        tutor_env_model.save()
        session.save()
        notebank.save()
        chat_history.save()
        db_manager = DatabaseManager(session.session_id)
        return session.session_id, db_manager

    @staticmethod
    def get_tutor_env(session_id: str) -> Tuple[str, "DatabaseManager"]:
        """Retrieves or creates a session environment based on the provided session UUID.

        Args:
            session_id (str): A string that potentially represents a valid UUID of a session.

        Returns:
            tuple: A tuple containing the session UUID and its corresponding DatabaseManager instance.
        """
        if type(session_id) is not str:
            session_id = ""

        try:
            # Try to fetch the session using the validated UUID
            _ = SessionModel.objects.get(session_id=session_id)
            return session_id, DatabaseManager(session_id)
        except SessionModel.DoesNotExist:
            # If the session does not exist, create a new one
            return DatabaseManager.create_tutor_session()

    def load_tutor_env(self) -> TutorEnv:
        # Load Chat History:
        chat_history_state = self.tutor_env_model.chat_history.chat

        # Load the Notebank:
        notebank_state = self.tutor_env_model.notebank.notes

        # Loads TutorEnv:
        self.tutor_env = TutorEnv.from_sql(
            self.tutor_env_model.curr_state, notebank_state, chat_history_state
        )

        # Load the ConceptDatabase associated with the TutorEnv:
        self.concept_database_model = self.tutor_env_model.concept_database
        if self.concept_database_model:
            concept_database_data = {
                "main_concept": self.concept_database_model.main_concept,
                "tutor_plan": self.tutor_env.notebank.env_string(),
                "concepts": self.concept_database_model.concepts_data,
            }
            self.concept_database = ConceptDatabase.from_sql(concept_database_data)

        # Load the SlidePlanner associated with the TutorEnv:
        self.slide_planner_model = self.tutor_env_model.slide_planner
        if self.slide_planner_model:
            slide_planner_data = {
                "current_obj_idx": self.slide_planner_model.current_obj_idx,
                "num_slides": self.slide_planner_model.num_slides,
                "slides_data": self.slide_planner_model.slides_data,
            }
            self.slide_planner = SlidePlanner.from_sql(
                slide_planner_data, self.tutor_env.notebank, self.concept_database
            )

        # Load the QuestionSuite associated with the TutorEnv:
        self.question_suite_model = self.tutor_env_model.question_suite
        if self.question_suite_model:
            self._num_questions = self.question_suite_model.num_questions
            self._qs_obj_idx = self.question_suite_model.current_obj_idx
            question_data = []
            for question in self.question_suite_model.questions.all():
                question_data.append(
                    [
                        question.subject,
                        question.type,
                        question.instructions,
                        question.data,
                        question.concepts.split("[SEP]"),
                    ]
                )

            self.question_suite = QuestionSuite.from_sql(
                self._qs_obj_idx,
                self._num_questions,
                question_data,
                self.tutor_env.notebank,
                self.concept_database,
            )

        # Load in Time-Dependent Features:
        if self.concept_database_model:
            self.tutor_env.concept_database = self.concept_database
        if self.question_suite_model:
            self.tutor_env.question_suite = self.question_suite
        if self.slide_planner_model:
            self.tutor_env.slide_planner = self.slide_planner

        # Return loaded instance of Tutor:
        return self.tutor_env

    def process_tutor_env(self, user_data):
        # Perform processing:
        system_data, current_state = self.tutor_env.step(user_data)
        return system_data, current_state

    def save_tutor_env(self):
        # Save any changes back to the database
        # This method should mirror the structure of the load_tutor_env method
        # but instead of loading, it should update the respective models with
        # the possibly changed data from the TutorEnv Python object

        # Save Notebank:
        self.tutor_env_model.notebank.notes = self.tutor_env.notebank.to_sql().strip()
        self.tutor_env_model.notebank.save()

        # Save Chat:
        self.tutor_env_model.chat_history.chat = self.tutor_env.chat_history.to_sql()
        self.tutor_env_model.chat_history.save()

        # Save Concept Model:
        if self.tutor_env.concept_database:
            concept_database_data = self.tutor_env.concept_database.to_sql()
            if not self.concept_database_model:  # Model was created on this time-step
                self.concept_database_model = ConceptDatabaseModel(
                    main_concept=concept_database_data["main_concept"],
                    concepts_data=concept_database_data["concepts"],
                )
                self.concept_database_model.save()
                self.tutor_env_model.concept_database = self.concept_database_model
                self.tutor_env_model.save()
            else:
                self.concept_database_model.main_concept = concept_database_data[
                    "main_concept"
                ]
                self.concept_database_model.concepts_data = concept_database_data[
                    "concepts"
                ]
                self.concept_database_model.save()

        # Save the Slide Planner:
        if self.tutor_env.slide_planner:
            slide_planner_data = self.tutor_env.slide_planner.to_sql()
            if not self.slide_planner_model:  # Model was created on this time-step
                self.slide_planner_model = SlidePlannerModel(
                    num_slides=slide_planner_data["num_slides"],
                    current_obj_idx=slide_planner_data["current_obj_idx"],
                    slides_data=slide_planner_data["slides_data"],
                )
                self.slide_planner_model.save()
                self.tutor_env_model.slide_planner = self.slide_planner_model
                self.tutor_env_model.save()
            else:
                self.slide_planner_model.num_slides = slide_planner_data["num_slides"]
                self.slide_planner_model.current_obj_idx = slide_planner_data[
                    "current_obj_idx"
                ]
                self.slide_planner_model.slides_data = slide_planner_data["slides_data"]
                self.slide_planner_model.save()

        # Save the Question Suite:
        if self.tutor_env.question_suite:
            qs_obj_idx, num_questions, questions = (
                self.tutor_env.question_suite.to_sql()
            )
            if not self.question_suite_model:  # Model was created on this time-step
                self.question_suite_model = QuestionSuiteModel(
                    num_questions=num_questions, current_obj_idx=qs_obj_idx
                )
                self.question_suite_model.save()
                # Link Concepts:
                question_models = []
                for question_data in questions:
                    question_id = uuid.uuid4()
                    question_model = QuestionModel.objects.create(
                        id=question_id,
                        subject=question_data[0],
                        type=question_data[1],
                        instructions=question_data[2],
                        data=question_data[3],
                        concepts="[SEP]".join(question_data[4]),
                    )
                    question_models += [question_model]
                # Link the concepts to the ConceptDatabaseModel:
                self.question_suite_model.questions.set(question_models)

                self.tutor_env_model.question_suite = self.question_suite_model
                self.tutor_env_model.save()
            self.question_suite_model.num_questions = (
                self.tutor_env.question_suite.num_questions
            )
            self.question_suite_model.current_obj_index = (
                self.tutor_env.question_suite.current_obj_idx
            )
            self.question_suite_model.save()

            question_models = []
            for question_data in questions:
                try:
                    # Attempt to get the question associated with the specific QuestionSuiteModel:
                    question_model = QuestionModel.objects.get(
                        instructions=question_data[2],
                        question_suite=self.question_suite_model,
                    )
                    # Update with any new data since you've found an existing model
                    question_model.subject = question_data[0]
                    question_model.type = question_data[1]
                    question_model.concepts = "[SEP]".join(question_data[4])
                    question_model.save()
                except QuestionModel.DoesNotExist:
                    # If it does not exist, create it and add it to the current QuestionSuiteModel:
                    question_model = QuestionModel.objects.create(
                        subject=question_data[0],
                        type=question_data[1],
                        instructions=question_data[2],
                        data=question_data[3],
                        concepts="[SEP]".join(question_data[4]),
                    )
                question_models.append(question_model)

            self.question_suite_model.questions.set(question_models)
            self.question_suite_model.current_obj_idx = (
                self.tutor_env.question_suite.current_obj_idx
            )
            self.tutor_env_model.question_suite = self.question_suite_model
            self.question_suite_model.save()
            self.tutor_env_model.save()

        # Update small Parameters:
        self.tutor_env_model.curr_state = self.tutor_env.current_state
        self.tutor_env_model.save()
        self.session.save()
