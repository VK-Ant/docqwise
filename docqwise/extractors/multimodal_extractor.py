"""Multimodal extraction — uses vision LLMs (Qwen2-VL, GPT-4o) for image-based documents.

For scanned documents, complex layouts, charts, handwriting — where text extraction fails.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Optional

from docqwise.core.document import DocqwiseDocument
from docqwise.core.field import FieldResult, ExtractionResult

logger = logging.getLogger("docqwise")


class MultimodalExtractor:
    """Extract fields from document images using vision LLMs.
    
    Supports:
    - OpenAI GPT-4o / GPT-4o-mini (API)
    - Google Gemini (API)
    - Qwen2-VL (local via Ollama or Transformers)
    - Any OpenAI-compatible vision endpoint
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None,
                 base_url: str = None):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

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
            response = self._call_vision_api(image_b64, extraction_prompt)
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

            response = self._call_vision_api(image_b64, prompt)
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
            response = self._call_vision_api(image_b64, prompt)
            return self._parse_response(response)
        except Exception as e:
            return {"error": str(e)}

    def _call_vision_api(self, image_b64: str, prompt: str) -> str:
        """Call vision API — supports OpenAI and compatible endpoints."""
        if "ollama" in self.model.lower() or self.base_url and "11434" in self.base_url:
            return self._call_ollama_vision(image_b64, prompt)
        else:
            return self._call_openai_vision(image_b64, prompt)

    def _call_openai_vision(self, image_b64: str, prompt: str) -> str:
        """Call OpenAI-compatible vision API."""
        import requests

        url = self.base_url or "https://api.openai.com/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

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

        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_ollama_vision(self, image_b64: str, prompt: str) -> str:
        """Call Ollama vision model (Qwen2-VL, LLaVA, etc.)."""
        import requests

        base = self.base_url or "http://localhost:11434"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        }
        resp = requests.post(f"{base}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "")

    def _encode_image(self, path: str) -> str:
        """Read and base64-encode an image file."""
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
