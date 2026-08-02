"""OpenAI LLM backend."""
from __future__ import annotations
import json
from docqwise.llm.base import BaseLLM

class OpenAILLM(BaseLLM):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        self.model = model
        self._api_key = api_key

    def generate(self, prompt: str, **kwargs) -> str:
        import openai
        client = openai.OpenAI(api_key=self._api_key)
        response = client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}],
            **kwargs)
        return response.choices[0].message.content

    def generate_structured(self, prompt: str, schema: dict) -> dict:
        full_prompt = f"{prompt}\n\nRespond ONLY with valid JSON:\n{json.dumps(schema)}"
        response = self.generate(full_prompt)
        try:
            clean = response.strip().strip("```json").strip("```").strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            return {"raw": response}
