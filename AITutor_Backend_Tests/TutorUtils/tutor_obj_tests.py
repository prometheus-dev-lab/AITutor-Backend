import os
import json
import unittest


from AITutor_Backend.src.TutorUtils.tutor_objs import (
    Chapter,
    Concept,
    ConceptDatabase,
    Lesson,
    NoteBank,
    TutorObjPrompts,
    TutorObjManager,
)

GENERATE_DATA = bool(os.environ.get("GENERATE_TESTS", 0))


c1 = Concept("Concept 1", None)
c2 = Concept("Concept 2", None)
cd = ConceptDatabase("Main Concept")
cd.Concepts += [c1, c2]

tutor_plan = """{"Notebank": [{"index": 0, "note": "User expresses interest in learning about agent AI."}
{"index": 1, "note": "Main Concept: Agent AI"}
{"index": 2, "note": "Student wants to learn about agent AI."}
{"index": 3, "note": "Subconcept: Introduction to Artificial Intelligence"}{"index": 4, "note": "Subconcept: Definition and Characteristics of Agents"}
{"index": 5, "note": "Subconcept: Agent Architectures"}
{"index": 6, "note": "Subconcept: Agent Environments"}
{"index": 7, "note": "Subconcept: Agent Communication and Coordination"}   
{"index": 8, "note": "Subconcept: Learning Agents and Adaptive Behavior"}  
{"index": 9, "note": "Subconcept: Multi-Agent Systems"}
{"index": 10, "note": "Subconcept: Intelligent Agents in Games, Robotics, and Simulation"}
{"index": 11, "note": "Subconcept: Ethical Considerations and Future Trends in Agent AI"}
{"index": 12, "note": "Tutor needs to gauge student's background knowledge 
in artificial intelligence and computer science."}
{"index": 13, "note": "Tutor should ask student about their specific interests in agent AI and any particular agent types or applications they want to learn about."}
{"index": 14, "note": "Tutor should inquire about the student's goals in learning about agent AI."}
{"index": 15, "note": "Tutor should ask student about their preference for 
a theoretical or practical approach to learning agent AI."}
{"index": 16, "note": "Tutor should ask student about their familiarity with programming languages or tools used in AI development."}
{"index": 17, "note": "Tutor to ask student about their familiarity with programming languages and tools used in AI development."}
{"index": 18, "note": "Tutor to ask student about their specific interests 
in agent AI and any particular agent types or applications they want to learn about."}
{"index": 19, "note": "Tutor to ask student about their goals in learning about agent AI."}
{"index": 20, "note": "Tutor to ask student about their preference for a theoretical or practical approach to learning agent AI."}
{"index": 21, "note": "Tutor should gauge student's current understanding of agent AI concepts to create a targeted learning plan."}
{"index": 22, "note": "Tutor should document their responses and preferences in the Notebank for future reference."}]}"""


class TutorObjManagerTests(unittest.TestCase):
    def setUp(self):
        self.notebank = NoteBank()
        self.concept_database = cd
        self.tutor_obj_manager = TutorObjManager(self.notebank, self.concept_database)

    def test_initialization(self):
        self.assertIsInstance(self.tutor_obj_manager.llm_prompts, TutorObjPrompts)
        self.assertEqual(self.tutor_obj_manager.Chapters, [])
        self.assertEqual(self.tutor_obj_manager.num_chapters, 0)
        self.assertFalse(self.tutor_obj_manager.initialized())

    def test_creation(
        self,
    ):
        if not GENERATE_DATA:
            return
        # TODO: Generation Tests
        notebank = NoteBank()
        j_data = json.loads(tutor_plan)

        [notebank.add_note(note["note"]) for note in j_data["Notebank"]]

        cd = ConceptDatabase("Agent AI", tutor_plan)
        cd.generate_concept_graph()

        tutor_obj_manager = TutorObjManager(notebank, cd)
        tutor_obj_manager.generate_chapters()

        self.assertIsNotNone(tutor_obj_manager.Chapters, "Could not generate chapters.")
        self.assertGreater(
            len((tutor_obj_manager.Chapters)), 0, "Could not create any Chapters."
        )
        self.assertIsInstance(tutor_obj_manager.Chapters[0], Chapter)
        tutor_obj_manager.generate_modules(0)


class ChapterTests(unittest.TestCase):
    def test_chapter_initialization(self):
        chapter = Chapter(
            "Test Chapter", "Overview", ["Outcome 1", "Outcome 2"], [c1, c2]
        )
        self.assertEqual(chapter.title, "Test Chapter")
        self.assertEqual(chapter.overview, "Overview")
        self.assertEqual(chapter.outcomes, ["Outcome 1", "Outcome 2"])
        self.assertEqual(chapter.concepts, [c1, c2])

    def test_create_chapters_from_JSON(self):
        llm_output = '{"Chapters": [{"title": "Chapter 1", "overview": "Overview 1", "outcomes": ["Outcome 1"], "concepts": ["Concept 1"]}, {"title": "Chapter 2", "overview": "Overview 2", "outcomes": ["Outcome 2"], "concepts": ["Concept 2"]}]}'
        success, chapters = Chapter.create_chapters_from_JSON(llm_output, cd)
        self.assertTrue(success)
        self.assertEqual(len(chapters), 2)
        self.assertIsInstance(chapters[0], Chapter)
        self.assertEqual(chapters[0].title, "Chapter 1")
        self.assertEqual(chapters[0].overview, "Overview 1")
        self.assertIn(
            "Outcome 1",
            chapters[0].outcomes,
        )
        self.assertIn(c1, chapters[0].concepts)

    def test_env_string(self):
        chapter = Chapter(
            "Test Chapter", "Overview", ["Outcome 1", "Outcome 2"], [c1, c2]
        )
        env_string = chapter.env_string()
        print(env_string)
        self.assertIn("Test Chapter", env_string)
        self.assertIn("Overview", env_string)
        self.assertIn("Outcome 1", env_string)
        self.assertIn("Concepts:", env_string)
        self.assertIn("Concept 1", env_string)


class LessonTests(unittest.TestCase):
    def test_lesson_initialization(self):
        lesson = Lesson(
            "Test Lesson",
            "Overview",
            ["Objective 1", "Objective 2"],
            [c1, c2],
            NoteBank.from_sql(tutor_plan),
            cd,
        )
        self.assertEqual(lesson.title, "Test Lesson")
        self.assertEqual(lesson.overview, "Overview")
        self.assertEqual(lesson.objectives, ["Objective 1", "Objective 2"])
        self.assertEqual(lesson.concepts, [c1, c2])

    def test_create_lessons_from_JSON(self):
        llm_output = '{"Lessons": [{"title": "Lesson 1", "overview": "Overview 1", "objectives": ["Objective 1"], "concepts": ["Concept 1"]}, {"title": "Lesson 2", "overview": "Overview 2", "objectives": ["Objective 2"], "concepts": ["Concept 2"]}]}'
        success, lessons = Lesson.create_lessons_from_JSON(
            llm_output, NoteBank.from_sql(tutor_plan), cd
        )
        self.assertTrue(success)
        self.assertEqual(len(lessons), 2)
        self.assertIsInstance(lessons[0], Lesson)
        self.assertEqual(lessons[0].title, "Lesson 1")

    def test_env_string(self):
        lesson = Lesson(
            "Test Lesson",
            "Overview",
            ["Objective 1", "Objective 2"],
            [c1, c2],
            NoteBank.from_sql(tutor_plan),
            cd,
        )
        env_string = lesson.env_string()
        self.assertIn("Test Lesson", env_string)
        self.assertIn("Overview", env_string)
        self.assertIn("Objective 1", env_string)
        self.assertIn("Concepts:", env_string)
        self.assertIn("Concept 1", env_string)
