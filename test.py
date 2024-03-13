import os

import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AITutor_Backend.settings')
django.setup()


import unittest

from AITutor_Backend_Tests import models_tests, tutor_env_tests
from AITutor_Backend_Tests.BackendUtils import code_executor_tests
from AITutor_Backend_Tests.TutorUtils import (chat_history_tests,
                                              concepts_tests, notebank_tests,
                                              prompts_tests, questions_tests,
                                              slides_tests, tutor_obj_tests)


def create_test_suite():
    test_suite = unittest.TestSuite()
    # Add Tests from Modules:
    test_suite.addTests(unittest.TestLoader().loadTestsFromModule(models_tests))
    test_suite.addTests(unittest.TestLoader().loadTestsFromModule(notebank_tests))
    test_suite.addTests(unittest.TestLoader().loadTestsFromModule(prompts_tests))
    test_suite.addTests(unittest.TestLoader().loadTestsFromModule(concepts_tests))
    test_suite.addTests(unittest.TestLoader().loadTestsFromModule(code_executor_tests))
    test_suite.addTests(unittest.TestLoader().loadTestsFromModule(chat_history_tests))
    test_suite.addTests(unittest.TestLoader().loadTestsFromModule(questions_tests))
    test_suite.addTests(unittest.TestLoader().loadTestsFromModule(slides_tests))
    test_suite.addTests(unittest.TestLoader().loadTestsFromModule(tutor_env_tests))
    test_suite.addTests(unittest.TestLoader().loadTestsFromModule(tutor_obj_tests))
    return test_suite

if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "AITutor_Backend.settings")
    unittest.TextTestRunner().run(create_test_suite())