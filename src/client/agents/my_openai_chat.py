import os
from typing import List, Dict

import requests

from src.client.agent import AgentClient


class MyOpenAIChat(AgentClient):
    def __init__(self, model: str = "gpt-4o-mini", api_base: str = "https://api.openai.com/v1",
                 api_key: str = None, temperature: float = 0.0, headers: Dict[str, str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required (env var or parameter).")
        self.temperature = temperature
        self.headers = headers or {}
        if "Authorization" not in self.headers:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        if "Content-Type" not in self.headers:
            self.headers["Content-Type"] = "application/json"

    def _convert_history(self, history: List[dict]) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        for item in history:
            role = "assistant" if item.get("role") == "agent" else "user"
            messages.append({"role": role, "content": item.get("content", "")})
        return messages

    def inference(self, history: List[dict]) -> str:
        messages = self._convert_history(history)
        url = f"{self.api_base}/chat/completions"
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature
        }
        resp = requests.post(url, json=body, headers=self.headers, timeout=120)
        if resp.status_code != 200:
            raise Exception(f"OpenAI API error {resp.status_code}: {resp.text}")
        data = resp.json()
        return data["choices"][0]["message"]["content"]


