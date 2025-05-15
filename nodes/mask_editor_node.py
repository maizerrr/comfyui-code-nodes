import comfy
import torch
from PIL import Image

class MaskEditorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
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
        
        # Add preview metadata for ComfyUI display
        preview_images = []
        for img_tensor in image:
            img = (img_tensor.cpu().numpy() * 255).astype('uint8')
            pil_img = Image.fromarray(img)
            preview_data = comfy.utils.save_temporary_image(pil_img, 'Preview_')
            preview_images.append({'filename': preview_data['filename'], 'subfolder': 'ComfyUI_temp', 'type': 'temp'})
        return {
            'result': (image, mask),
            'ui': {
                'images': preview_images
            }
        }

NODE_CLASS_MAPPINGS = {"MaskEditorNode": MaskEditorNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskEditorNode": "Mask Editor"}