import json
from typing import List, Tuple, Union
import re
from enum import IntEnum
import threading
import openai
from AITutor_Backend.src.BackendUtils.json_serialize import *
from AITutor_Backend.src.BackendUtils.json_serialize import JSONSerializable
from AITutor_Backend.src.TutorUtils.concepts import *
from AITutor_Backend.src.DataUtils.file_utils import save_training_data
from enum import IntEnum
from typing import List, Tuple

import os
DEBUG = bool(os.environ.get("DEBUG", 0))

class Chapter:
    def __init__(self, overview, outcomes, concepts):
        self.Lessons = []
        self.overview = overview
        self.outcomes = outcomes
        self.concepts = concepts
    
    def generate_lesson_plan(self, ):
        pass

class Lesson:
    def __init__(self, title, overview, objectives, concepts):
        self.Slides = []
        self.Homework = []
        self.title = title
        self. overview = overview
        self.objectives = objectives
        self.concepts = concepts