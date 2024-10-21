CONFIG_MANAGER = {
    # Slides:
    "planSlides": "AITutor_Backend/src/TutorUtils/Prompts/KnowledgePhase/Slides/slide_plan_prompt",
    "objConvertSlides": "AITutor_Backend/src/TutorUtils/Prompts/KnowledgePhase/Slides/slide_obj_prompt",
    "dialogueSlides": "AITutor_Backend/src/TutorUtils/Prompts/KnowledgePhase/Slides/slide_dialogue_prompt",
    
    # Questions:
    "planQuestions": "AITutor_Backend/src/TutorUtils/Prompts/KnowledgePhase/Questions/plan_question_prompt",
    "objConvertQuestions": "AITutor_Backend/src/TutorUtils/Prompts/KnowledgePhase/Questions/question_obj_prompt",
    "questionData": "AITutor_Backend/src/TutorUtils/Prompts/KnowledgePhase/Questions/question_data_prompt",
    
    # Tutor Objs:
    "planChapters": "AITutor_Backend/src/TutorUtils/Prompts/KnowledgePhase/TutorObjs/chapter_plan_prompt",
    "objConvertChapters": "AITutor_Backend/src/TutorUtils/Prompts/KnowledgePhase/TutorObjs/chapter_obj_prompt",
    "planLessons": "AITutor_Backend/src/TutorUtils/Prompts/KnowledgePhase/TutorObjs/lesson_plan_prompt",
    "objConvertLessons": "AITutor_Backend/src/TutorUtils/Prompts/KnowledgePhase/TutorObjs/lesson_obj_prompt",
    
    # NoteBank:
    "noteGeneration": "AITutor_Backend/src/TutorUtils/Prompts/notebank_prompt",
    
    # Concept Graph:
    "conceptGraph": "AITutor_Backend/src/TutorUtils/Prompts/KnowledgePhase/Concepts/concept_graph_prompt",
    "conceptPrompt": "AITutor_Backend/src/TutorUtils/Prompts/KnowledgePhase/Concepts/concept_prompt",
}


def get_prompt_config(config_name: str):
    if config_name.startswith("@") and config_name[1:] in CONFIG_MANAGER:
        return CONFIG_MANAGER[config_name[1:]]
    else:
        return config_name
