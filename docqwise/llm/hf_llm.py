"""HuggingFace Transformers LLM backend — runs models locally via transformers library."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from docqwise.llm.base import BaseLLM

logger = logging.getLogger("docqwise")


class HuggingFaceLLM(BaseLLM):
    """Run HuggingFace models locally via transformers."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        quantize: str = "4bit",
        device: str = "auto",
        max_new_tokens: int = 2000,
    ):
        self.model_name = model_name
        self.quantize = quantize
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model: {self.model_name} (quantize={self.quantize})")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True,
        )

        model_kwargs = {"trust_remote_code": True}

        # Dtype
        if torch.cuda.is_available():
            model_kwargs["dtype"] = torch.float16
        else:
            model_kwargs["dtype"] = torch.float32

        # Quantization
        if self.quantize == "4bit" and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs["device_map"] = "auto"
            except ImportError:
                model_kwargs["device_map"] = "auto"
        elif self.quantize == "8bit" and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs["device_map"] = "auto"
            except ImportError:
                model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = "auto" if torch.cuda.is_available() else "cpu"

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs,
        )

        logger.info(f"Model loaded: {self.model_name}")

    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        self._ensure_loaded()

        max_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temperature = kwargs.get("temperature", 0.1)

        # Build chat messages
        sys_msg = system_prompt or "You are a document extraction AI. Always respond with valid JSON only. No explanation. No markdown."
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ]

        # Apply chat template
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        inputs = self._tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._model.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._model.device)

        import torch
        with torch.no_grad():
            outputs = self._model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the NEW tokens (not the prompt)
        new_tokens = outputs[0][input_ids.shape[1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response.strip()

    def generate_structured(self, prompt: str, schema: dict) -> dict:
        full_prompt = (
            f"{prompt}\n\n"
            f"Return ONLY valid JSON. No explanation:\n"
            f"{json.dumps(schema, indent=2)}"
        )
        response = self.generate(full_prompt)

        # Parse JSON
        text = response.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            return {"raw": response}

    def unload(self):
        """Free GPU memory."""
        import gc
        self._model = None
        self._tokenizer = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
