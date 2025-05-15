import comfy
import openai

class OpenAIQueryNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (["gpt-3.5-turbo", "gpt-4o"], {"default": "gpt-4o"}),
                "system_prompt": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "user_query": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "process"
    CATEGORY = "custom_code"

    def process(self, api_key, model, system_prompt, user_query):
        if not api_key:
            raise ValueError("API key is required")

        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ]
            )
            return (response.choices[0].message.content,)
        except Exception as e:
            return (f"Error: {str(e)}",)

NODE_CLASS_MAPPINGS = {"OpenAIQueryNode": OpenAIQueryNode}
NODE_DISPLAY_NAME_MAPPINGS = {"OpenAIQueryNode": "OpenAI Chat Query"}