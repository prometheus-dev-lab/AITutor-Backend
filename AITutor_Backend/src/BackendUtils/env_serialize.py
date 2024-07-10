class EnvSerializable:
    """
    A class instance is used as input to a Large Language Model (LLM).
    """

    def __init__(self):
        pass

    def env_string(
        self,
    ):
        """
        Returns a Markdown Representation of the Class for a Language Model to interpret.
        """
        raise NotImplementedError()
