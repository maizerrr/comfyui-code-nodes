import openai
import base64
import traceback
import json
from PIL import Image
import io
import torch
import numpy as np
from typing import List

def base64_to_tensor(base64_image: str) -> torch.Tensor:
    image_bytes = base64.b64decode(base64_image)
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor_image = torch.from_numpy(np.array(pil_image)).float().div(255)
    return tensor_image

def tensor_to_files(tensor: torch.Tensor, is_mask: bool=False) -> List[io.BytesIO]:
    files = []
    prefix = "mask" if is_mask else "image"
    suffix = "png" if is_mask else "jpg"
    formatting = "PNG" if is_mask else "JPEG"
    channel = "L" if is_mask else "RGB"
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    for i in range (tensor.size(0)):
        t = tensor[i].cpu().float()
        if t.size(0) == 3:
            t = t.permute(1, 2, 0)
        if t.size(0) == 1:
            t = t.squeeze(0)
        pil_image = Image.fromarray((t.numpy() * 255).astype('uint8'), channel)
        buffered = io.BytesIO()
        pil_image.save(buffered, format=formatting)
        buffered.name = f"{prefix}-{i}.{suffix}"
        files.append(buffered)
    return files

class OpenAIGPTImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "user_query": ("STRING", {"multiline": True}),
                "size": (["1024x1024", "1536x1024", "1024x1536"], {"default": "1024x1024"}),
                "quality": (["high", "medium", "low", "auto"], {"default": "high"}),
                "background": (["auto", "transparent", "opaque"], {"default": "auto"}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10}),
            },
            "optional": {
                "reference_images": ("IMAGE",),
                "masks": ("MASK",)
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("gpt_image", "raw_response")
    FUNCTION = "process"
    CATEGORY = "custom_code"

    def process(self, api_key: str, user_query: str, size: str, quality: str, n: int, reference_images: torch.Tensor=None, masks: torch.Tensor=None, **kwargs):
        if not api_key:
            raise ValueError("API key is required")
        try:
            client = openai.OpenAI(api_key=api_key)
            image_files = []
            if reference_images is not None:
                image_files = tensor_to_files(reference_images, is_mask=False)
            mask_files = []
            if masks is not None:
                mask_files = tensor_to_files(masks, is_mask=True)
            if len(mask_files) == 1 and len(image_files) == 1:
                # currently only support single image-mask pair
                response = client.images.edit(
                    model="gpt-image-1",
                    image=image_files[0],
                    mask=mask_files[0],
                    prompt=user_query,
                    size=size,
                    quality=quality,
                    n=n
                )
            elif len(image_files) != 0:
                response = client.images.edit(
                    model="gpt-image-1",
                    image=image_files,
                    prompt=user_query,
                    size=size,
                    quality=quality,
                    n=n
                )
            else:
                response = client.images.generate(
                    model="gpt-image-1",
                    prompt=user_query,
                    size=size,
                    quality=quality,
                    n=n
                )
            for f in image_files + mask_files:
                f.close()
            if not hasattr(response, "data") or not response.data or not hasattr(response.data[0], "b64_json"):
                raise RuntimeError(f"Invalid response from OpenAI API: missing image data. Full response: {response}")
            results = []
            for d in response.data:
                if not hasattr(d, "b64_json"):
                    continue
                base64_image = d.b64_json
                results.append(base64_to_tensor(base64_image))
            results = torch.stack(results, dim=0)
            return (results, "Debug info")
        except Exception:
            raise RuntimeError(f"OpenAI GPT-Image-1 Node Error: {traceback.format_exc()}")

NODE_CLASS_MAPPINGS = {"OpenAIGPTImageNode": OpenAIGPTImageNode}
NODE_DISPLAY_NAME_MAPPINGS = {"OpenAIGPTImageNode": "OpenAI GPT-Image-1 Node"}