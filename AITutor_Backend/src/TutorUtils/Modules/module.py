import os
import re
from enum import IntEnum
from typing import List, Tuple

from AITutor_Backend.src.BackendUtils.json_serialize import JSONSerializable
from AITutor_Backend.src.BackendUtils.sql_serialize import SQLSerializable


class Module:
    pass # TODO: A module is a Lesson, an Exam, a Quiz, ecetera