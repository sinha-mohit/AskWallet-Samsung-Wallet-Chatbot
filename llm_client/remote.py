import requests
import logging
from .base import LLMClient

class RemoteHuggingFaceClient(LLMClient):
    def __init__(self, model_id: str, api_url: str, token: str):
        self.model_id = model_id
        self.api_url = api_url
        self.headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def generate(self, messages):
        payload = {
            "model": self.model_id, "stream": False, "max_tokens": 1024, "temperature": 0.3, "top_p": 0.9,
            "messages": messages,
            "stop": None
        }
        logging.info(f"Sending payload to REMOTE LLM API: {payload}")
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            logging.info(f"API response: {response.json()}")
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            raise RuntimeError("Request to the language model timed out.")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")
