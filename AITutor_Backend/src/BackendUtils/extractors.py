import re
import json


class JSONExtractor:
    # Regular expression to find possible JSON structures, handling both objects and arrays
    JSON_REGEX = re.compile(r"(\{.*?\}|\[.*?\])", re.DOTALL)

    @staticmethod
    def extract(data: str) -> dict | list | None:
        # Preprocess the data to remove any bad characters
        data = data.replace("\'s", "s")
        data = data.replace("\n", "")

        # Find the first position of either a '[' or '{'
        start_index = min(
            (data.find("{") if "{" in data else float("inf")),
            (data.find("[") if "[" in data else float("inf")),
        )

        # Find the last position of either a '}' or ']'
        end_index = max(
            (data.rfind("}") if "}" in data else -float("inf")),
            (data.rfind("]") if "]" in data else -float("inf")),
        )

        # If no valid JSON start or end character is found, return None
        if (
            start_index == float("inf")
            or end_index == -float("inf")
            or start_index > end_index
        ):
            return None

        # Extract the substring between the first opening bracket and last closing bracket
        possible_json_str = data[start_index : end_index + 1]

        try:
            json_obj = json.loads(possible_json_str)
            return json_obj
        except json.JSONDecodeError:
            pass
        

        # Find all potential JSON matches (objects or arrays) within the limited substring
        matches = JSONExtractor.JSON_REGEX.findall(possible_json_str)

        # Attempt to find the largest valid JSON structure
        largest_json_obj = None
        for match in matches:
            try:
                # Try loading the current match as a JSON object
                json_obj = json.loads(match)

                # Keep track of the largest valid JSON object/array
                if largest_json_obj is None or len(match) > len(
                    json.dumps(largest_json_obj)
                ):
                    largest_json_obj = json_obj
            except json.JSONDecodeError:
                # If this match is not valid JSON, continue with the next match
                continue

        # Return the largest valid JSON object/array, or None if none found
        return largest_json_obj

data = """
```json
{
        "Chapters": [
                    {
                                    "title": "Introduction to Agent AI",
                                    "overview": "This chapter serves as an introductory guide to the world of Agent AI, exploring the foundational concepts critical to understanding the field. Students will learn about intelligence, agency, autonomy, and rationality—concepts that underpin the functioning and design of intelligent agents.",
                                    "outcomes": "After studying this chapter, students should be able to define and explain the key fundamental concepts of Agent AI, including intelligence, agency, autonomy, and rationality, and understand their interrelations and importance in the design of AI systems.",
                                    "concepts": [
                                                        "Intelligence",
                                                        "Agency",
                                                        "Autonomy",
                                                        "Rationality"
                                                ]
                    },
                    {
                                    "title": "Types of Agents",
                                    "overview": "This chapter categorizes the different types of agents used in AI, from simple reflex agents to more complex learning agents. Students will understand how agents differ in capabilities and functions and their suitability for various tasks.",
                                    "outcomes": "By the end of this chapter, students should be able to identify and differentiate between various types of agents, describing the characteristics and practical uses of each type.",
                                    "concepts": [
                                        "Simple Reflex Agents",
                                        "Model-Based Reflex Agents",
                                        "Goal-Based Agents",
                                        "Utility-Based Agents",
                                        "Learning Agents"
                                        ]
                    },
                    {
                                    "title": "Agent Environments",
                                    "overview": "Delve into the various environments in which agents operate. This chapter explains the attributes of these environments and how they influence agent design and functionality.",
                                    "outcomes": "Students should be able to classify and describe different agent environments, understanding the implications of these environments on agent behavior and performance.",
                                    "concepts": [    "Fully Observable vs. Partially Observable",    "Deterministic vs. Stochastic",    "Episodic vs. Sequential",    "Static vs. Dynamic",    "Discrete vs. Continuous",    "Single-Agent vs. Multi-Agent"]
                    },
                    {
                                    "title": "Agent Architecture",
                                    "overview": "Understanding the architecture of agents is crucial. This chapter covers how perception, decision-making, and action frameworks combine to form the architecture of an agent.",
                                    "outcomes": "Students should be able to explain the components of agent architecture and how they function together to create effective AI systems.",
                                    "concepts": [    "Perception",    "Decision Making",    "Action"]
                    },
                    {
                                    "title": "Learning in Agents",
                                    "overview": "This chapter introduces the learning methods applicable to agents, discussing how agents can adapt and improve using different types of learning processes such as supervised, unsupervised, and reinforcement learning.",
                                    "outcomes": "Students should be able to compare and contrast the different learning methods used in AI and understand their applications in developing adaptive and intelligent agents.",
                                    "concepts": [    "Supervised Learning",    "Unsupervised Learning",    "Reinforcement Learning"]
                    },
                    {
                                    "title": "Agent Programming",
                                    "overview": "Explore the programming languages and frameworks essential for developing agents. This chapter examines the tools and technologies used to build and implement intelligent agents.",
                                    "outcomes": "Upon completion, students should be proficient in identifying appropriate programming languages and frameworks for agent development and understand their roles in building AI applications.",
                                    "concepts": [    "Languages (e.g., Python, Java)",    "Frameworks (e.g., TensorFlow, PyTorch)"]
                    },
                    {
                                    "title": "Applications of Agent AI",
                                    "overview": "This chapter explores the practical applications of agent AI across various fields such as robotics, natural language processing, and autonomous vehicles.",
                                    "outcomes": "Students should be able to identify and describe different real-world applications of agent AI and the significance of agents in these contexts.",
                                    "concepts": [    "Robotics",    "Natural Language Processing",    "Computer Vision",    "Game AI",    "Autonomous Vehicles"]
                    },
                    {
                                    "title": "Ethical Considerations in AI",
                                    "overview": "Ethics play a critical role in AI development. This chapter covers the ethical considerations in AI, focusing on safety, bias, and governance issues.",
                                    "outcomes": "Students should be able to articulate the ethical challenges AI poses, including safety concerns, bias, and governance, and propose approaches to address these issues.",
                                    "concepts": [    "AI Safety",    "Bias in AI",    "AI Governance"]
                    },
                    {
                                    "title": "Related AI Concepts",
                                    "overview": "Broaden the horizon with an exploration of related AI concepts that complement the study of agent AI, including machine learning and neural networks, among others.",
                                    "outcomes": "Students should gain insight into how related AI concepts like machine learning and neural networks integrate with agent AI, understanding their roles and impacts on the development of intelligent systems.",
                                    "concepts": [    "Machine Learning",    "Neural Networks",    "Expert Systems",    "Fuzzy Logic",    "Genetic Algorithms"]
                    },
                    {
                                    "title": "Philosophical Aspects of AI",
                                    "overview": "Engage in philosophical discussions concerning AI\'s nature and capabilities, exploring concepts such as weak AI vs. strong AI, the Turing test, and the possibility of Artificial General Intelligence (AGI).",
                                    "outcomes": "Students should be able to engage in critical discussions about the philosophical implications and future possibilities of AI, including the distinctions between weak and strong AI and the concept of AGI.",
                                    "concepts": [    "Weak AI vs. Strong AI",    "The Turing Test",    "Artificial General Intelligence (AGI)"]
                    }
        ]
}
```"""

print(JSONExtractor.extract(data))