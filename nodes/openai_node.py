import comfy
import openai
import torch
from PIL import Image
import io
import base64
from typing import List

class OpenAIQueryNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (["gpt-3.5-turbo", "gpt-4o"], {"default": "gpt-4o"}),
                "system_prompt": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "user_query": ("STRING", {"multiline": True}),
            },
            "optional": {
                "image_input": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "encoded_image")
    FUNCTION = "process"
    CATEGORY = "custom_code"

    def process(self, api_key: str, model: str, system_prompt: str, user_query: str, image_input: List[torch.Tensor]=None) -> tuple[str, str]:
        if not api_key:
            raise ValueError("API key is required")

        try:
            client = openai.OpenAI(api_key=api_key)
            encoded_images = self.process_image(image_input) if image_input is not None else []
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_query},
                        *([{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}} 
                        for base64_image in encoded_images])
                    ]}
                ]
            )
            return (response.choices[0].message.content, ", ".join(encoded_images))
        except Exception as e:
            return (f"Error: {str(e)}", "")

    def process_image(self, image_tensor: List[torch.Tensor]) -> List[str]:
        if image_tensor is None:
            return []
        
        images = []
        for tensor in image_tensor:
            # Convert tensor to PIL Image
            tensor = tensor.cpu().float()
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            # Handle channel-first format if needed
            if tensor.size(0) == 3:
                tensor = tensor.permute(1, 2, 0)
            pil_image = Image.fromarray((tensor.numpy() * 255).astype('uint8'), 'RGB')
            
            # Convert to JPEG base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            images.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
        
        return images

NODE_CLASS_MAPPINGS = {"OpenAIQueryNode": OpenAIQueryNode}
NODE_DISPLAY_NAME_MAPPINGS = {"OpenAIQueryNode": "GhatGPT Node"}