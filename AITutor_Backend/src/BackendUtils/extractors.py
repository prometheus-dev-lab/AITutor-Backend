import re
import json


class JSONExtractor:
    JSON_REGEX = re.compile(r"\`\`\`json([^\`]*)\`\`\`")

    @staticmethod
    def extract(data: str) -> dict | None:
        # Search string for JSON object
        match = JSONExtractor.JSON_REGEX.findall(data)
        if match:
            match = match[0].replace("```json", "").replace("```", "").strip()

        # Use the match, otherwise try to use the data
        json_data = match if match else data
        json_obj = json.loads(json_data)  # Throws JSON Decode errror if bad data

        # Successfully extracted the obj; return the value
        return json_obj
