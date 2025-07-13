import requests
import logging
from .base import LLMClient

class LocalLLMClient(LLMClient):
    def __init__(self, model_id: str, api_url: str):
        self.model_id = model_id
        self.api_url = api_url
        self.headers = {"Content-Type": "application/json"}

    def generate(self, messages):
        system_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_and_assistant = "\n\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages if m["role"] != "system"])
        full_prompt = f"{system_prompt}\n\n{user_and_assistant}"
        payload = {
            "model": self.model_id,
            "prompt": full_prompt,
            "max_tokens": 1024,
            "temperature": 0.3,
            "top_p": 0.9,
            "stop": None,
            "stream": False
        }
        logging.info(f"Sending payload to LOCAL LLM API: {payload}")
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()["content"]
        except requests.exceptions.Timeout:
            raise RuntimeError("Request to the local language model timed out.")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Local API request failed: {e}")
