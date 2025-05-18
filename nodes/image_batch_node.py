import torch
import comfy
from typing import List, Optional

class ImageBatchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
            },
            "optional": {
                "image2": ("IMAGE", { "default": None }),
                "image3": ("IMAGE", { "default": None }),
                "image4": ("IMAGE", { "default": None }),
                "image5": ("IMAGE", { "default": None }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("batched_image",)
    FUNCTION = "batch_images"
    CATEGORY = "custom_code"

    def batch_images(self, image1: torch.Tensor, 
                     image2: Optional[torch.Tensor] = None, 
                     image3: Optional[torch.Tensor] = None, 
                     image4: Optional[torch.Tensor] = None, 
                     image5: Optional[torch.Tensor] = None) -> tuple[torch.Tensor]:
        
        images_to_combine = [image1]
        
        if image2 is not None:
            images_to_combine.append(image2)
        if image3 is not None:
            images_to_combine.append(image3)
        if image4 is not None:
            images_to_combine.append(image4)
        if image5 is not None:
            images_to_combine.append(image5)
        
        if not images_to_combine:
            # Should not happen due to required image1, but as a safeguard
            return (torch.empty(0),)

        # Ensure all tensors have the same H, W, C dimensions
        # B_ixHxWx3
        first_image_shape = images_to_combine[0].shape
        h, w, c = first_image_shape[1], first_image_shape[2], first_image_shape[3]

        processed_images = []
        for img_tensor in images_to_combine:
            if img_tensor.shape[1:] != (h, w, c):
                # This case should ideally be handled by ComfyUI or raise an error
                # For now, we'll skip tensors with mismatched dimensions to avoid crashing
                # Or, one could attempt to resize, but that's beyond the current scope
                print(f"Skipping image with shape {img_tensor.shape} due to dimension mismatch with first image shape {first_image_shape}")
                continue
            processed_images.append(img_tensor)
        
        if not processed_images:
             return (torch.empty(0),) # Or raise error

        batched_image = torch.cat(processed_images, dim=0)
        return (batched_image,)

NODE_CLASS_MAPPINGS = {
    "ImageBatchNode": ImageBatchNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBatchNode": "Batch Images (up to 5)"
}