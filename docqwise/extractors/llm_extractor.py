"""LLM-based field extraction — uses any LLM for intelligent document understanding.

Supports: Ollama, HuggingFace Transformers, OpenAI, Anthropic.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from docqwise.core.document import DocqwiseDocument
from docqwise.core.field import FieldResult, ExtractionResult
from docqwise.extractors.base import BaseFieldExtractor

logger = logging.getLogger("docqwise")


class LLMFieldExtractor(BaseFieldExtractor):
    """Extract fields using LLM. Falls back to regex only if no LLM available."""

    def __init__(self, llm=None, model: str = None):
        self._llm = llm
        self._model = model

    def _get_llm(self):
        if self._llm is not None:
            return self._llm

        # Priority 1: Ollama
        try:
            from urllib.request import urlopen
            urlopen("http://localhost:11434/api/tags", timeout=2)
            from docqwise.llm.ollama import OllamaLLM
            model = self._model or "nemotron-mini"
            self._llm = OllamaLLM(model=model)
            return self._llm
        except Exception:
            pass

        # Priority 2: HuggingFace
        try:
            from docqwise.llm.hf_llm import HuggingFaceLLM
            import torch
            if torch.cuda.is_available():
                model = self._model or "Qwen/Qwen2.5-7B-Instruct"
                self._llm = HuggingFaceLLM(model_name=model, quantize="4bit")
                return self._llm
        except ImportError:
            pass

        # Priority 3: OpenAI
        try:
            import os
            if os.environ.get("OPENAI_API_KEY"):
                from docqwise.llm.openai_llm import OpenAILLM
                self._llm = OpenAILLM(model=self._model or "gpt-4o-mini")
                return self._llm
        except Exception:
            pass

        return None

    def extract_fields(self, document: DocqwiseDocument,
                       schema: dict = None, prompt: str = None,
                       prompt_template: str = None, system_prompt: str = None) -> ExtractionResult:
        llm = self._get_llm()

        if llm is None:
            from docqwise.extractors.field_extractor import RegexFieldExtractor
            return RegexFieldExtractor().extract_fields(document, schema=schema)

        text = document.text

        # User prompt takes priority
        if prompt:
            if "{context}" in prompt:
                final_prompt = prompt.replace("{context}", text)
            else:
                final_prompt = f"{prompt}\n\nDocument:\n{text}"
            response = llm.generate(final_prompt)
            fields = self._parse_response_to_fields(response, schema)
            confidence = sum(f.confidence for f in fields.values()) / max(len(fields), 1)
            result = ExtractionResult(
                doc_id=document.doc_id, source_path=document.source_path,
                fields=fields, confidence=confidence, strategy_used="llm",
            )
        elif prompt_template:
            schema_str = json.dumps(schema, indent=2) if schema else ""
            final_prompt = prompt_template.replace("{context}", text).replace("{schema}", schema_str)
            response = llm.generate(final_prompt)
            fields = self._parse_response_to_fields(response, schema)
            confidence = sum(f.confidence for f in fields.values()) / max(len(fields), 1)
            result = ExtractionResult(
                doc_id=document.doc_id, source_path=document.source_path,
                fields=fields, confidence=confidence, strategy_used="llm",
            )
        elif schema:
            result = self._extract_with_schema(llm, document, text, schema)
        else:
            result = self._extract_auto(llm, document, text)

        # If LLM returned 0 fields, try regex silently
        if len(result.fields) == 0:
            from docqwise.extractors.field_extractor import RegexFieldExtractor
            result = RegexFieldExtractor().extract_fields(document, schema=schema)
            result.strategy_used = "regex"

        return result

    def _extract_with_schema(self, llm, document, text, schema):
        field_list = []
        for name, spec in schema.items():
            desc = spec.get("description", name.replace("_", " "))
            ftype = spec.get("type", "string")
            field_list.append(f"  {name} ({ftype}): {desc}")
        fields_text = "\n".join(field_list)

        prompt = (
            f"Extract these fields from the document below.\n\n"
            f"Fields:\n{fields_text}\n\n"
            f"Document:\n{text}\n\n"
            f"Return the result as a JSON object with field names as keys. "
            f"For numbers remove currency symbols and commas. "
            f"For dates use YYYY-MM-DD. If not found set null.\n"
            f"JSON:"
        )

        response = llm.generate(prompt)
        logger.debug(f"LLM raw response:\n{response[:1000]}")
        fields = self._parse_response_to_fields(response, schema)

        confidence = sum(f.confidence for f in fields.values()) / max(len(fields), 1)
        return ExtractionResult(
            doc_id=document.doc_id, source_path=document.source_path,
            fields=fields, confidence=confidence, strategy_used="llm",
        )

    def _extract_auto(self, llm, document, text):
        prompt = (
            f"Read this document and extract all key information as key-value pairs.\n\n"
            f"Document:\n{text}\n\n"
            f"Return a JSON object with descriptive field names as keys. "
            f"Numbers without currency symbols. Dates as YYYY-MM-DD.\n"
            f"JSON:"
        )

        response = llm.generate(prompt)
        logger.debug(f"LLM raw response:\n{response[:1000]}")
        fields = self._parse_response_to_fields(response)

        confidence = sum(f.confidence for f in fields.values()) / max(len(fields), 1)
        return ExtractionResult(
            doc_id=document.doc_id, source_path=document.source_path,
            fields=fields, confidence=confidence, strategy_used="llm",
        )

    def _parse_response_to_fields(self, response: str, schema: dict = None) -> dict[str, FieldResult]:
        """Parse LLM response into fields. Tries multiple strategies."""
        fields = {}

        # Strategy 1: Parse as JSON
        parsed = self._try_parse_json(response)
        if parsed:
            fields = self._dict_to_fields(parsed, schema)
            if fields:
                return fields

        # Strategy 2: Extract key:value pairs from text
        fields = self._extract_kv_from_text(response, schema)
        if fields:
            return fields

        # Strategy 3: If schema provided, search for values in response
        if schema:
            fields = self._search_schema_values(response, schema)
            if fields:
                return fields

        return fields

    def _try_parse_json(self, response: str) -> dict | None:
        """Try multiple ways to parse JSON from response."""
        text = response.strip()

        # Remove markdown
        text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
        text = text.strip()

        # Direct parse
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Find outermost { ... } using brace counting
        brace_depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == '{':
                if brace_depth == 0:
                    start = i
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1
                if brace_depth == 0 and start >= 0:
                    candidate = text[start:i + 1]
                    try:
                        result = json.loads(candidate)
                        if isinstance(result, dict):
                            return result
                    except json.JSONDecodeError:
                        # Try fixing common issues
                        fixed = self._fix_json(candidate)
                        if fixed:
                            return fixed
                    break

        return None

    def _fix_json(self, text: str) -> dict | None:
        """Fix common JSON issues from LLMs."""
        # Remove trailing commas
        fixed = re.sub(r',\s*}', '}', text)
        fixed = re.sub(r',\s*]', ']', fixed)

        # Fix single quotes to double quotes
        fixed = fixed.replace("'", '"')

        try:
            result = json.loads(fixed)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        return None

    def _extract_kv_from_text(self, response: str, schema: dict = None) -> dict[str, FieldResult]:
        """Extract key:value or key=value pairs from plain text response."""
        fields = {}
        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip().lstrip('- ').lstrip('* ')
            if not line:
                continue

            # Match patterns: "key: value", "key = value", "key - value"
            match = re.match(r'^["\']?([a-zA-Z_][a-zA-Z0-9_ ]*)["\']?\s*[:=\-]\s*["\']?(.+?)["\']?\s*$', line)
            if match:
                key = match.group(1).strip().lower().replace(' ', '_')
                value = match.group(2).strip().rstrip(',')

                # Skip if key looks like explanation text
                if len(key) > 30 or key in ('note', 'explanation', 'comment', 'summary'):
                    continue

                # Type the value
                typed_value = self._auto_type(value)

                if schema:
                    # Only include if key matches schema
                    for schema_key in schema:
                        if key == schema_key or key.replace('_', '') == schema_key.replace('_', ''):
                            fields[schema_key] = FieldResult(
                                value=typed_value, confidence=0.88,
                                raw_text=value, extraction_method="llm",
                            )
                            break
                else:
                    fields[key] = FieldResult(
                        value=typed_value, confidence=0.85,
                        raw_text=value, extraction_method="llm",
                    )

        return fields

    def _search_schema_values(self, response: str, schema: dict) -> dict[str, FieldResult]:
        """Search for schema field values in response text."""
        fields = {}
        text_lower = response.lower()

        for field_name, spec in schema.items():
            readable = field_name.replace('_', ' ')
            desc = spec.get('description', readable)

            # Search for "field_name: value" or "description: value"
            for search_term in [field_name, readable, desc.split('.')[0].lower()]:
                pattern = rf'{re.escape(search_term)}\s*[:=\-]\s*["\']?(.+?)["\',\n}}\]]'
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    if value and value.lower() not in ('null', 'none', 'n/a', 'not found'):
                        typed_value = self._auto_type(value)
                        field_type = spec.get('type', 'string')
                        if field_type in ('number', 'integer'):
                            typed_value = self._to_number(value)
                        fields[field_name] = FieldResult(
                            value=typed_value, confidence=0.85,
                            raw_text=value, extraction_method="llm",
                        )
                        break

        return fields

    def _dict_to_fields(self, data: dict, schema: dict = None) -> dict[str, FieldResult]:
        """Convert parsed dict to FieldResult dict."""
        fields = {}
        # Flatten nested dicts
        flat = self._flatten_dict(data)

        for key, value in flat.items():
            if key in ('document_type', 'doc_type', 'type'):
                continue
            if value is None or str(value).lower() in ('null', 'none', 'n/a', ''):
                continue

            # Type conversion if schema available
            if schema and key in schema:
                field_type = schema[key].get('type', 'string')
                if field_type in ('number', 'integer'):
                    value = self._to_number(value)

            fields[key] = FieldResult(
                value=value, confidence=0.92,
                raw_text=str(value), extraction_method="llm",
            )

        return fields

    def _flatten_dict(self, d: dict, parent_key: str = '') -> dict:
        """Flatten nested dict: {"a": {"b": 1}} -> {"a_b": 1}."""
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}_{k}" if parent_key else k
            if isinstance(v, dict) and not any(isinstance(vv, (dict, list)) for vv in v.values()):
                items.update(self._flatten_dict(v, new_key))
            else:
                items[new_key] = v
        return items

    def _auto_type(self, value: str) -> Any:
        """Auto-detect and convert type."""
        if not isinstance(value, str):
            return value
        num = self._to_number(value)
        if num is not None:
            return num
        if value.lower() in ('true', 'yes'):
            return True
        if value.lower() in ('false', 'no'):
            return False
        return value

    def _to_number(self, value: Any) -> float | None:
        """Try to convert to number."""
        if isinstance(value, (int, float)):
            return value
        try:
            cleaned = re.sub(r'[,$\s]', '', str(value))
            cleaned = re.sub(r'(?:USD|INR|EUR|GBP|Rs\.?|₹|%)', '', cleaned).strip()
            if cleaned:
                return float(cleaned)
        except (ValueError, TypeError):
            pass
        return None


class LLMEntityExtractor:
    """Extract entities using LLM."""

    def __init__(self, llm=None):
        self._llm = llm

    def _get_llm(self):
        if self._llm:
            return self._llm
        extractor = LLMFieldExtractor()
        return extractor._get_llm()

    def extract_entities(self, document: DocqwiseDocument,
                         types: list[str] = None,
                         custom_types: dict = None) -> list:
        from docqwise.core.element import Entity

        llm = self._get_llm()
        if llm is None:
            from docqwise.extractors.entity_extractor import RegexEntityExtractor
            return RegexEntityExtractor().extract_entities(document, types=types)

        text = document.text
        type_instruction = ""
        if types:
            type_instruction = f"Extract only these types: {', '.join(types)}\n"
        if custom_types:
            type_instruction += f"Also extract: {json.dumps(custom_types)}\n"

        prompt = (
            f"Extract all named entities from this document.\n"
            f"{type_instruction}\n"
            f"Document:\n{text}\n\n"
            f"Return a JSON array. Each item: {{\"text\": \"entity\", \"type\": \"TYPE\"}}\n"
            f"Types: PERSON, ORG, MONEY, DATE, LOCATION, EMAIL, PHONE\n"
            f"JSON:"
        )

        try:
            response = llm.generate(prompt)
            logger.debug(f"Entity LLM response:\n{response[:500]}")

            entities = []
            seen = set()

            # Try JSON array parse
            parsed = self._parse_array(response)
            for item in parsed:
                if isinstance(item, dict) and "text" in item and "type" in item:
                    key = (item["text"], item["type"])
                    if key not in seen:
                        seen.add(key)
                        entities.append(Entity(
                            text=item["text"], entity_type=item["type"], confidence=0.88,
                        ))

            # If JSON parse got results, return them
            if entities:
                return entities

            # Otherwise extract from text response
            for line in response.split('\n'):
                line = line.strip().lstrip('- *')
                match = re.match(r'\[?(\w+)\]?\s*[:=\-]\s*(.+)', line)
                if match:
                    etype = match.group(1).upper()
                    etext = match.group(2).strip().strip('"\'')
                    if len(etext) < 50:
                        key = (etext, etype)
                        if key not in seen:
                            seen.add(key)
                            entities.append(Entity(text=etext, entity_type=etype, confidence=0.85))

            if entities:
                return entities

            # Final fallback to regex
            from docqwise.extractors.entity_extractor import RegexEntityExtractor
            return RegexEntityExtractor().extract_entities(document, types=types)

        except Exception:
            from docqwise.extractors.entity_extractor import RegexEntityExtractor
            return RegexEntityExtractor().extract_entities(document, types=types)

    def _parse_array(self, response: str) -> list:
        text = response.strip()
        text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
        text = text.strip()

        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
            if isinstance(result, dict) and "entities" in result:
                return result["entities"]
        except json.JSONDecodeError:
            pass

        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                # Try fixing
                candidate = text[start:end]
                candidate = re.sub(r',\s*]', ']', candidate)
                candidate = candidate.replace("'", '"')
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
        return []
