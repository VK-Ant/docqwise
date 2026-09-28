"""Ollama LLM backend — zero external dependencies (uses urllib)."""
from __future__ import annotations
import json
from docqwise.llm.base import BaseLLM

class OllamaLLM(BaseLLM):
    def __init__(self, model: str = "qwen2.5:7b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, **kwargs) -> str:
        import urllib.request
        system_prompt = kwargs.get("system_prompt")
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if system_prompt:
            payload["system"] = system_prompt
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode()).get("response", "")

    def generate_structured(self, prompt: str, schema: dict) -> dict:
        full_prompt = f"{prompt}\n\nRespond ONLY with valid JSON matching this schema:\n{json.dumps(schema)}"
        response = self.generate(full_prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"raw": response}
