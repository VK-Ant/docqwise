"""Ollama LLM backend."""
from __future__ import annotations
import json
from docqwise.llm.base import BaseLLM


class OllamaLLM(BaseLLM):
    def __init__(self, model: str = "qwen2.5:7b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        from urllib.request import urlopen, Request
        import json as json_mod
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt
        req = Request(
            f"{self.base_url}/api/generate",
            data=json_mod.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = urlopen(req, timeout=120)
        return json_mod.loads(resp.read()).get("response", "")

    def generate_structured(self, prompt: str, schema: dict, system_prompt: str = None) -> dict:
        full_prompt = f"{prompt}\n\nRespond ONLY with valid JSON matching this schema:\n{json.dumps(schema)}"
        response = self.generate(full_prompt, system_prompt=system_prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"raw": response}
