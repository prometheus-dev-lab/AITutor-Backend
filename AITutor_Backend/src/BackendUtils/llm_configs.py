LLM_CONFIG_MANAGER = {
    # Slides:
    "planSlides": "gpt-4o",
    "objConvertSlides": "gpt-4o-mini",
    "dialogueSlides": "gpt-4o-mini",
    
    # Questions:
    "planQuestions": "claude-3-5-sonnet-20240620",
    "objConvertQuestions": "gpt-4o-mini",
    "questionData": "claude-3-5-sonnet-20240620",
    
    # Tutor Objs:
    "planChapters": "gpt-4o",
    "objConvertChapters": "gpt-4o-mini",
    "planLessons": "gpt-4o",
    "objConvertLessons": "gpt-4o-mini",
    
    # NoteBank:
    "noteGeneration": "gpt-4o-mini",

    # Concept Graph:
    "conceptGraph": "claude-3-5-sonnet-20240620",
    "conceptPrompt": "gpt-4o-mini",
}


def get_llm_config(config_name: str):
    if config_name.startswith("@") and config_name[1:] in LLM_CONFIG_MANAGER:
        return LLM_CONFIG_MANAGER[config_name[1:]]
    else:
        return config_name
