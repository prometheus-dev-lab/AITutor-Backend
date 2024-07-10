import unittest

from AITutor_Backend.src.tutor_env import TutorEnv
from AITutor_Backend.src.TutorUtils.concepts import Concept, ConceptDatabase
from AITutor_Backend.src.TutorUtils.Modules.questions import Question, QuestionSuite
from AITutor_Backend.src.TutorUtils.Modules.slides import Slide, SlidePlanner


class TutorEnvTests(unittest.TestCase):
    def setUp(self):
        self.tutor_env = TutorEnv()

    def test_populate_data_from_file(self):
        # Call the method under test
        self.tutor_env = TutorEnv.populate_data_from_file()
        
        # Basic assertions to ensure proper loading: We dont care about the data as long as it exists
        self.assertIsNotNone(
            self.tutor_env.concept_database, "Concepts should not be None"
        )
        
        self.assertIsNotNone(
            self.tutor_env.obj_manager, "Object Manager should not be None"
        )
        
        self.assertGreater(
            self.tutor_env.obj_manager.num_chapters, 0, "Chapters should be greater than 0."
        )

        self.assertGreater(
            self.tutor_env.obj_manager.num_chapters, 0, "Chapters should be greater than 0."
        )

        
        self.assertNotEqual(
            self.tutor_env.obj_manager.curr_chapter_idx,
            -1,
            "Unexpected number of slides loaded",
        )


        self.assertNotEqual(
            self.tutor_env.obj_manager.curr_lesson_idx,
            -1,
            "Unexpected number of slides loaded",
        )
