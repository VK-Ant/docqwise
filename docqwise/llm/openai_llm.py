"""OpenAI LLM backend — supports OpenAI, Azure, vLLM, LM Studio, any compatible API.

Uses urllib (no requests dependency) for Python 3.9-3.14 compatibility.
"""
from __future__ import annotations
import json
import os
import urllib.request
from docqwise.llm.base import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None,
                 base_url: str = None):
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or "https://api.openai.com/v1"

    def generate(self, prompt: str, **kwargs) -> str:
        system_prompt = kwargs.pop("system_prompt", None)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 2000),
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode())
        return result["choices"][0]["message"]["content"]

    def generate_structured(self, prompt: str, schema: dict) -> dict:
        full_prompt = f"{prompt}\n\nRespond ONLY with valid JSON:\n{json.dumps(schema)}"
        response = self.generate(full_prompt)
        try:
            clean = response.strip().strip("```json").strip("```").strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            return {"raw": response}
