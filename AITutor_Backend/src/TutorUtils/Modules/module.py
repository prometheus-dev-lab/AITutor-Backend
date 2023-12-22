from enum import IntEnum
from typing import Tuple, List
import re
import os
from AITutor_Backend.src.BackendUtils.sql_serialize import SQLSerializable
from AITutor_Backend.src.BackendUtils.json_serialize import JSONSerializable

class Module:
    pass # TODO: A module is a Lesson, an Exam, a Quiz, ecetera