import os
import json
from typing import List, Dict, Any

import requests

from src.client.agent import AgentClient


def _to_gemini_contents(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	contents: List[Dict[str, Any]] = []
	for m in messages or []:
		role = m.get("role", "user")
		text = m.get("content", "") or ""
		parts: List[Dict[str, Any]] = []
		if isinstance(text, str) and text:
			parts.append({"text": text})
		# Best-effort pass-through for tool responses if present in messages
		# Some controllers inject tool replies as role "tool" with plain content
		if role == "tool" and not parts and text:
			parts.append({"text": text})
		if role == "assistant":
			role = "model"
		contents.append({"role": role, "parts": parts or [{"text": ""}]})
	return contents


def _to_gemini_function_declarations(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	funcs: List[Dict[str, Any]] = []
	for t in tools or []:
		fn = t.get("function", {})
		if not fn:
			continue
		funcs.append(
			{
				"name": fn.get("name", ""),
				"description": fn.get("description", ""),
				"parameters": fn.get("parameters", {}),  # pass JSON schema through
			}
		)
	return funcs


class MyGeminiChat(AgentClient):
	def __init__(self, model: str = None, api_key: str = None, api_base: str = None, temperature: float = 0.0, headers: Dict[str, str] = None, *args, **kwargs):
		super().__init__(*args, **kwargs)
		# Allow overrides via env vars; default to v1 + modern model
		self.model = model or os.getenv("GEMINI_MODEL") or "gemini-2.5-pro"
		self.api_key = api_key or os.getenv("GEMINI_API_KEY")
		if not self.api_key:
			raise ValueError("GEMINI_API_KEY is required (env var or parameter).")
		self.api_base = (api_base or os.getenv("GEMINI_API_BASE") or "https://generativelanguage.googleapis.com/v1").rstrip("/")
		self.temperature = temperature
		self.headers = headers or {"Content-Type": "application/json"}

	def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
		url = f"{self.api_base}/models/{self.model}:{path}?key={self.api_key}"
		resp = requests.post(url, headers=self.headers, data=json.dumps(body), timeout=120)
		if resp.status_code != 200:
			raise Exception(f"Gemini API error {resp.status_code}: {resp.text}")
		return resp.json()

	def inference(self, history: List[dict]) -> str:
		contents = _to_gemini_contents(history or [])
		body = {
			"contents": contents,
			"generationConfig": {"temperature": self.temperature},
		}
		data = self._post("generateContent", body)
		candidates = (data or {}).get("candidates") or []
		if not candidates:
			return ""
		parts = candidates[0].get("content", {}).get("parts") or []
		for p in parts:
			if "text" in p:
				return p["text"]
		return ""

	def inference_with_tools(self, *, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
		contents = _to_gemini_contents(messages or [])
		function_declarations = _to_gemini_function_declarations(tools or [])
		body: Dict[str, Any] = {
			"contents": contents,
			"tools": [{"functionDeclarations": function_declarations}] if function_declarations else [],
			"toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
			"generationConfig": {"temperature": self.temperature},
		}
		data = self._post("generateContent", body)
		candidates = (data or {}).get("candidates") or []
		if not candidates:
			return {"role": "assistant", "content": ""}
		parts = candidates[0].get("content", {}).get("parts") or []
		# Map Gemini functionCall -> OpenAI-style tool_calls
		tool_calls: List[Dict[str, Any]] = []
		accumulated_text: List[str] = []
		for p in parts:
			if "functionCall" in p:
				fc = p["functionCall"]
				name = fc.get("name", "")
				args = fc.get("args", {})
				tool_calls.append(
					{
						"id": f"call_{name}",
						"type": "function",
						"function": {
							"name": name,
							"arguments": json.dumps(args, ensure_ascii=False),
						},
					}
				)
			elif "text" in p:
				accumulated_text.append(p["text"])
		assistant_message: Dict[str, Any] = {"role": "assistant"}
		if tool_calls:
			assistant_message["tool_calls"] = tool_calls
			assistant_message["content"] = None
		else:
			assistant_message["content"] = "\n".join(accumulated_text).strip()
		return assistant_message


