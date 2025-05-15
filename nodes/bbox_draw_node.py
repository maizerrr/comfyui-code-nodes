import comfy
import torch
from PIL import Image, ImageDraw
import colorsys
import numpy as np

class BBoxDrawNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bboxes": ("BBOX",),
            },
            "optional": {
                "color": ("STRING", {"default": "#FF0000"}),
                "line_width": ("INT", {"default": 2, "min": 1, "max": 10}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "draw_boxes"
    CATEGORY = "custom_code"

    def draw_boxes(self, image, bboxes, color="#FF0000", line_width=2):
        batch_size, height, width, channels = image.shape
        result = torch.zeros_like(image)
        
        for i in range(batch_size):
            img = image[i].numpy()
            pil_img = Image.fromarray((img * 255).astype('uint8'))
            draw = ImageDraw.Draw(pil_img)
            
            try:
                for (x1, y1, x2, y2) in bboxes:
                    draw.rectangle([(x1, y1), (x2, y2)], 
                                  outline=color, 
                                  width=line_width)
            except (ValueError, TypeError) as e:
                pass
            
            result[i] = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0)
        
        return (result,)

NODE_CLASS_MAPPINGS = {"BBoxDrawNode": BBoxDrawNode}
NODE_DISPLAY_NAME_MAPPINGS = {"BBoxDrawNode": "BBox Drawer"}