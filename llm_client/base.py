from abc import ABC, abstractmethod
from typing import List, Dict

class LLMClient(ABC):
    @abstractmethod
    def generate(self, messages: List[Dict]) -> str:
        pass
