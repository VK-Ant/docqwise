"""OpenAI LLM backend — works with OpenAI, Azure, vLLM, LM Studio, any compatible API."""
from __future__ import annotations
import json
from docqwise.llm.base import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None,
                 base_url: str = None):
        self.model = model
        self._api_key = api_key
        self._base_url = base_url

    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        from urllib.request import urlopen, Request
        import os

        url = self._base_url or "https://api.openai.com/v1/chat/completions"
        if not url.endswith("/chat/completions"):
            url = url.rstrip("/") + "/chat/completions"

        api_key = self._api_key or os.environ.get("OPENAI_API_KEY", "")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 2000),
        }

        req = Request(url, data=json.dumps(payload).encode(), headers=headers)
        resp = urlopen(req, timeout=120)
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    def generate_structured(self, prompt: str, schema: dict, system_prompt: str = None) -> dict:
        full_prompt = f"{prompt}\n\nRespond ONLY with valid JSON:\n{json.dumps(schema)}"
        response = self.generate(full_prompt, system_prompt=system_prompt)
        try:
            clean = response.strip().strip("```json").strip("```").strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            return {"raw": response}
