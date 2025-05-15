import comfy
import re
from typing import List, Tuple

class BBoxParseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_input": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("BBOX",)
    RETURN_NAMES = ("bboxes",)
    FUNCTION = "parse"
    CATEGORY = "custom_code"

    def parse(self, text_input: str) -> List[Tuple[int, int, int, int]]:
        try:
            matches = re.findall(r'\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)', text_input)
            boxes = []
            for m in matches:
                boxes.append(tuple(int(num) for num in m))
            return (boxes,)
        except (ValueError, TypeError) as e:
            return ([],)

NODE_CLASS_MAPPINGS = {"BBoxParseNode": BBoxParseNode}
NODE_DISPLAY_NAME_MAPPINGS = {"BBoxParseNode": "BBox Parser"}