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
    CATEGORY = "custom_code"

    def process(self, **kwargs):
        # Get the actual input type if connected
        input_data = kwargs.get("any_input")
        if input_data is not None:
            self.RETURN_TYPES = (type(input_data).__name__,)
        return (kwargs.get("any_input"),)

NODE_CLASS_MAPPINGS = {"DummyNode": DummyNode}
NODE_DISPLAY_NAME_MAPPINGS = {"DummyNode": "Dummy Passthrough Node"}