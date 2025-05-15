import comfy
import torch

class MaskEditorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"display": "image", "type": "image_upload"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "process"
    CATEGORY = "custom_code"
    OUTPUT_NODE = True

    def process(self, image):
        batch_size, height, width, _ = image.shape
        mask = torch.zeros((batch_size, height, width), dtype=torch.float32)
        return (image, mask)

NODE_CLASS_MAPPINGS = {"MaskEditorNode": MaskEditorNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskEditorNode": "Mask Editor"}