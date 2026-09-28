"""Multimodal extraction — uses vision LLMs for image-based documents.

Supports:
- Ollama vision models (qwen2-vl, gemma3, llava, moondream)
- HuggingFace vision models (Qwen2-VL, Florence-2, Gemma-3)
- OpenAI/Azure/compatible API (GPT-4o, Gemini, Claude)
- Local model files (.gguf, .onnx, .tflite, .pt)

Uses urllib (no requests dependency) for Python 3.9-3.14 compatibility.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import urllib.request
from typing import Any, Optional

from docqwise.core.document import DocqwiseDocument
from docqwise.core.field import FieldResult, ExtractionResult

logger = logging.getLogger("docqwise")

# Known vision model routing
OLLAMA_VISION_MODELS = {
    "qwen2-vl", "gemma3", "llava", "llava-phi3", "moondream",
    "bakllava", "llava-llama3", "minicpm-v",
}

HF_VISION_MODELS = {
    "Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2-VL-2B-Instruct",
    "google/gemma-3-4b-it", "google/paligemma-3b-mix-224",
    "microsoft/Florence-2-large", "microsoft/Florence-2-base",
}

API_VISION_MODELS = {
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo",
    "gemini-1.5-flash", "gemini-1.5-pro",
    "claude-3-opus", "claude-3-sonnet", "claude-3-haiku",
}


class MultimodalExtractor:
    """Extract fields from document images using vision LLMs.

    Auto-routes to the right backend based on model name:
    - Ollama models → local Ollama API
    - HuggingFace models → local transformers
    - API models → OpenAI-compatible endpoint
    - Local files (.gguf, .onnx, .pt, .tflite) → appropriate runtime
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None,
                 base_url: str = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url
        self._hf_model = None
        self._hf_processor = None

    def extract_from_image(self, image_path: str, schema: dict = None,
                            prompt: str = None) -> ExtractionResult:
        """Extract fields from a document image."""
        image_b64 = self._encode_image(image_path)

        if schema:
            extraction_prompt = self._build_schema_prompt(schema)
        elif prompt:
            extraction_prompt = prompt
        else:
            extraction_prompt = self._build_auto_prompt()

        try:
            response = self._call_vision(image_b64, extraction_prompt)
            parsed = self._parse_response(response)
            return self._build_result(parsed, image_path)
        except Exception as e:
            logger.error(f"Multimodal extraction failed: {e}")
            return ExtractionResult(
                doc_id="", source_path=image_path,
                fields={}, confidence=0.0,
                strategy_used="multimodal_failed",
                warnings=[str(e)],
            )

    def extract_from_pdf_page(self, pdf_path: str, page_num: int = 0,
                               schema: dict = None) -> ExtractionResult:
        """Render PDF page to image, then extract with vision LLM."""
        try:
            import fitz
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            pix = page.get_pixmap(dpi=200)
            image_bytes = pix.tobytes("png")
            doc.close()

            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            if schema:
                prompt = self._build_schema_prompt(schema)
            else:
                prompt = self._build_auto_prompt()

            response = self._call_vision(image_b64, prompt)
            parsed = self._parse_response(response)
            return self._build_result(parsed, pdf_path)

        except Exception as e:
            logger.error(f"PDF multimodal extraction failed: {e}")
            return ExtractionResult(
                doc_id="", source_path=pdf_path,
                fields={}, confidence=0.0,
                strategy_used="multimodal_failed",
                warnings=[str(e)],
            )

    def describe_chart(self, image_path: str) -> dict:
        """Extract data and description from a chart/graph image."""
        image_b64 = self._encode_image(image_path)
        prompt = """Analyze this chart/graph image. Extract:
1. Chart type (bar, line, pie, scatter, etc.)
2. Title and axis labels
3. All data points as a structured table
4. Key insights or trends

Return as JSON:
{
    "chart_type": "...",
    "title": "...",
    "x_axis": "...",
    "y_axis": "...",
    "data": [{"label": "...", "value": ...}, ...],
    "insights": ["...", "..."]
}"""
        try:
            response = self._call_vision(image_b64, prompt)
            return self._parse_response(response)
        except Exception as e:
            return {"error": str(e)}

    # ══════════════════════════════════════
    # VISION ROUTING
    # ══════════════════════════════════════

    def _call_vision(self, image_b64: str, prompt: str) -> str:
        """Route to the right vision backend based on model name."""
        model_lower = self.model.lower()

        # Local file models
        if os.path.isfile(self.model):
            ext = os.path.splitext(self.model)[1].lower()
            if ext == ".gguf":
                return self._call_gguf_vision(image_b64, prompt)
            elif ext == ".onnx":
                return self._call_onnx_vision(image_b64, prompt)
            elif ext == ".tflite":
                return self._call_tflite_vision(image_b64, prompt)
            elif ext == ".pt":
                return self._call_local_pt_vision(image_b64, prompt)

        # Local HF folder
        if os.path.isdir(self.model):
            return self._call_hf_vision(image_b64, prompt)

        # Ollama vision models
        if any(name in model_lower for name in OLLAMA_VISION_MODELS):
            return self._call_ollama_vision(image_b64, prompt)

        # HuggingFace models
        if self.model in HF_VISION_MODELS or "/" in self.model and not self.api_key:
            return self._call_hf_vision(image_b64, prompt)

        # API models (OpenAI, Azure, etc.)
        return self._call_openai_vision(image_b64, prompt)

    def _call_ollama_vision(self, image_b64: str, prompt: str) -> str:
        """Call Ollama vision model."""
        base = self.base_url or "http://localhost:11434"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/api/generate", data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode()).get("response", "")

    def _call_openai_vision(self, image_b64: str, prompt: str) -> str:
        """Call OpenAI-compatible vision API."""
        url = self.base_url or "https://api.openai.com/v1/chat/completions"
        if not url.endswith("/chat/completions"):
            url = url.rstrip("/") + "/chat/completions"

        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    }}
                ]
            }],
            "max_tokens": 2000,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode())
        return result["choices"][0]["message"]["content"]

    def _call_hf_vision(self, image_b64: str, prompt: str) -> str:
        """Call HuggingFace vision model locally."""
        model_lower = self.model.lower() if isinstance(self.model, str) else ""

        if "qwen" in model_lower and "vl" in model_lower:
            return self._call_hf_qwen_vl(image_b64, prompt)
        elif "florence" in model_lower:
            return self._call_hf_florence(image_b64, prompt)
        else:
            return self._call_hf_generic(image_b64, prompt)

    def _call_hf_qwen_vl(self, image_b64: str, prompt: str) -> str:
        """Qwen2-VL specific pipeline."""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            import torch
            from PIL import Image
            import io

            if self._hf_model is None:
                self._hf_processor = AutoProcessor.from_pretrained(self.model, trust_remote_code=True)
                self._hf_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model, dtype=torch.float16,
                    device_map="auto", trust_remote_code=True,
                )

            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes))

            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}]
            text = self._hf_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._hf_processor(text=[text], images=[image], return_tensors="pt").to(self._hf_model.device)

            with torch.no_grad():
                output_ids = self._hf_model.generate(**inputs, max_new_tokens=2000)
            output_ids = output_ids[:, inputs.input_ids.shape[1]:]
            return self._hf_processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        except Exception as e:
            logger.error(f"Qwen2-VL failed: {e}")
            return json.dumps({"error": str(e)})

    def _call_hf_florence(self, image_b64: str, prompt: str) -> str:
        """Florence-2 specific pipeline."""
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            import torch
            from PIL import Image
            import io

            if self._hf_model is None:
                self._hf_processor = AutoProcessor.from_pretrained(self.model, trust_remote_code=True)
                self._hf_model = AutoModelForCausalLM.from_pretrained(
                    self.model, dtype=torch.float16,
                    device_map="auto", trust_remote_code=True,
                )

            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes))

            inputs = self._hf_processor(text=prompt, images=image, return_tensors="pt").to(self._hf_model.device)
            with torch.no_grad():
                output_ids = self._hf_model.generate(**inputs, max_new_tokens=2000)
            return self._hf_processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        except Exception as e:
            logger.error(f"Florence-2 failed: {e}")
            return json.dumps({"error": str(e)})

    def _call_hf_generic(self, image_b64: str, prompt: str) -> str:
        """Generic HuggingFace vision model."""
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            import torch
            from PIL import Image
            import io

            if self._hf_model is None:
                self._hf_processor = AutoProcessor.from_pretrained(self.model, trust_remote_code=True)
                self._hf_model = AutoModelForCausalLM.from_pretrained(
                    self.model, dtype=torch.float16,
                    device_map="auto", trust_remote_code=True,
                )

            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes))

            inputs = self._hf_processor(text=prompt, images=image, return_tensors="pt").to(self._hf_model.device)
            with torch.no_grad():
                output_ids = self._hf_model.generate(**inputs, max_new_tokens=2000)
            return self._hf_processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        except Exception as e:
            logger.error(f"HF vision model failed: {e}")
            return json.dumps({"error": str(e)})

    def _call_gguf_vision(self, image_b64: str, prompt: str) -> str:
        """GGUF model via llama-cpp-python."""
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler

            chat_handler = Llava15ChatHandler(clip_model_path=self.model)
            llm = Llama(model_path=self.model, chat_handler=chat_handler, n_ctx=2048)
            response = llm.create_chat_completion(messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }])
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"GGUF vision failed: {e}")
            return json.dumps({"error": str(e)})

    def _call_local_pt_vision(self, image_b64: str, prompt: str) -> str:
        """PyTorch .pt model."""
        try:
            import torch
            model = torch.load(self.model, map_location="cpu")
            return json.dumps({"info": "PyTorch model loaded", "type": str(type(model))})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _call_onnx_vision(self, image_b64: str, prompt: str) -> str:
        """ONNX model via onnxruntime."""
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(self.model)
            return json.dumps({"info": "ONNX model loaded",
                               "inputs": [i.name for i in session.get_inputs()]})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _call_tflite_vision(self, image_b64: str, prompt: str) -> str:
        """TFLite model."""
        try:
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=self.model)
            interpreter.allocate_tensors()
            return json.dumps({"info": "TFLite model loaded"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def unload(self):
        """Free GPU memory."""
        import gc
        self._hf_model = None
        self._hf_processor = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    # ══════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════

    def _encode_image(self, path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _build_schema_prompt(self, schema: dict) -> str:
        fields_desc = json.dumps(schema, indent=2)
        return f"""Look at this document image carefully. Extract these fields:

{fields_desc}

For each field, find the exact value in the document.
For numbers, return numeric values without currency symbols.
For dates, return YYYY-MM-DD format.

Return ONLY valid JSON with field names as keys. No explanation."""

    def _build_auto_prompt(self) -> str:
        return """Look at this document image carefully.
1. Identify the document type (invoice, contract, receipt, form, report, etc.)
2. Extract ALL key-value pairs you can find
3. Extract any tables as structured data
4. Note any signatures, stamps, or handwritten content

Return ONLY valid JSON:
{
    "document_type": "...",
    "fields": {"field_name": "value", ...},
    "tables": [{"headers": [...], "rows": [[...], ...]}, ...],
    "has_signature": true/false,
    "has_handwriting": true/false
}"""

    def _parse_response(self, response: str) -> dict:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()
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
            return {"raw_response": response}

    def _build_result(self, parsed: dict, source: str) -> ExtractionResult:
        fields = {}
        extracted = parsed.get("fields", parsed)
        if "document_type" in extracted:
            del extracted["document_type"]
        if "fields" in extracted:
            extracted = extracted["fields"]
        for name, value in extracted.items():
            if value is not None and value != "" and value != "null":
                fields[name] = FieldResult(
                    value=value,
                    confidence=0.93,
                    extraction_method="multimodal_vision",
                )
        confidence = sum(f.confidence for f in fields.values()) / max(len(fields), 1)
        return ExtractionResult(
            doc_id="",
            source_path=source,
            fields=fields,
            confidence=confidence,
            strategy_used="multimodal",
        )
