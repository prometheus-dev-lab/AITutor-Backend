import re
import json


class JSONExtractor:
    # Regular expression to find possible JSON structures, handling both objects and arrays
    JSON_REGEX = re.compile(r"(\{.*?\}|\[.*?\])", re.DOTALL)

    @staticmethod
    def extract(data: str) -> dict | list | None:
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
