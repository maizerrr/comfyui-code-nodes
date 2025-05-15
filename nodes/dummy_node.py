import comfy

class DummyNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {"any_input": ("*",)},
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "process"
    CATEGORY = "utils"

    def process(self, **kwargs):
        return (kwargs.get("any_input"),)

NODE_CLASS_MAPPINGS = {"DummyNode": DummyNode}
NODE_DISPLAY_NAME_MAPPINGS = {"DummyNode": "Dummy Passthrough Node"}