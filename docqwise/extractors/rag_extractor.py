"""RAG-based document extraction — the core of docqwise.

Instead of truncating text and hoping LLM catches everything:
1. Read full document (any length)
2. Chunk with structure preservation
3. Embed all chunks
4. For each field: retrieve the most relevant chunks
5. Send focused context to LLM
6. Extract with precision

Works on 1-page invoices and 1000-page contracts equally well.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from docqwise.core.document import DocqwiseDocument
from docqwise.core.chunk import DocqwiseChunk
from docqwise.core.field import FieldResult, ExtractionResult

logger = logging.getLogger("docqwise")


class RAGExtractor:
    """RAG-based field extraction. No truncation. Full document intelligence.
    
    Usage:
        from docqwise.extractors.rag_extractor import RAGExtractor
        
        extractor = RAGExtractor(llm=my_llm, embedder=my_embedder)
        result = extractor.extract_fields(document, schema=invoice_schema)
    """

    def __init__(self, llm=None, embedder=None, chunker=None,
                 chunk_top_k: int = 5, model: str = None):
        self._llm = llm
        self._embedder = embedder
        self._chunker = chunker
        self._chunk_top_k = chunk_top_k
        self._model = model

    def _ensure_llm(self):
        if self._llm is not None:
            return self._llm
        # Auto-detect available LLM
        try:
            import requests
            requests.get("http://localhost:11434/api/tags", timeout=2)
            from docqwise.llm.ollama import OllamaLLM
            self._llm = OllamaLLM(model=self._model or "nemotron-mini")
            return self._llm
        except Exception:
            pass
        try:
            from docqwise.llm.hf_llm import HuggingFaceLLM
            import torch
            if torch.cuda.is_available():
                self._llm = HuggingFaceLLM(
                    model_name=self._model or "Qwen/Qwen2.5-3B-Instruct",
                    quantize="4bit",
                )
                return self._llm
        except ImportError:
            pass
        return None

    def _ensure_embedder(self):
        if self._embedder is not None:
            return self._embedder
        try:
            from docqwise.embedders.sentence_transformer import SentenceTransformerEmbedder
            self._embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
            return self._embedder
        except ImportError:
            return None

    def _ensure_chunker(self):
        if self._chunker is not None:
            return self._chunker
        from docqwise.chunkers.structure_chunker import StructureChunker
        self._chunker = StructureChunker(max_tokens=300, overlap=50)
        return self._chunker

    def extract_fields(self, document: DocqwiseDocument,
                       schema: dict = None, prompt: str = None,
                       prompt_template: str = None) -> ExtractionResult:
        """Extract fields using RAG pipeline. Full document, no truncation.

        Args:
            document: parsed document
            schema: field definitions (optional)
            prompt: user's complete prompt — use {context} placeholder for document text
            prompt_template: user's prompt template — use {context} and {schema} placeholders
        """
        llm = self._ensure_llm()
        if llm is None:
            logger.info("No LLM available for RAG extraction")
            from docqwise.extractors.field_extractor import RegexFieldExtractor
            return RegexFieldExtractor().extract_fields(document, schema=schema)

        embedder = self._ensure_embedder()
        chunker = self._ensure_chunker()

        # Step 1: Chunk the full document
        chunks = chunker.chunk(document)
        logger.info(f"Document chunked: {len(chunks)} chunks from {document.metadata.word_count} words")

        if not chunks:
            return ExtractionResult(
                doc_id=document.doc_id, source_path=document.source_path,
                fields={}, confidence=0.0, strategy_used="rag",
            )

        # Step 2: Embed all chunks
        if embedder:
            import numpy as np
            chunk_texts = [c.text for c in chunks]
            embeddings = embedder.embed_batch(chunk_texts)
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb

        # Step 3: Extract — user prompt takes priority
        if prompt:
            context = "\n\n".join([c.text for c in chunks])
            fields = self._extract_with_user_prompt(llm, document, prompt, context)
        elif prompt_template:
            fields = self._extract_with_user_template(
                llm, document, chunks, schema, embedder, prompt_template,
            )
        elif schema:
            fields = self._extract_with_schema_rag(llm, document, chunks, schema, embedder)
        else:
            fields = self._extract_auto_rag(llm, document, chunks, embedder)

        confidence = sum(f.confidence for f in fields.values()) / max(len(fields), 1)
        return ExtractionResult(
            doc_id=document.doc_id, source_path=document.source_path,
            fields=fields, confidence=confidence, strategy_used="rag",
        )

    def _extract_with_user_prompt(self, llm, document, prompt, context):
        """User provided their own complete prompt."""
        if "{context}" in prompt:
            final_prompt = prompt.replace("{context}", context)
        else:
            final_prompt = f"{prompt}\n\nDocument:\n{context}"
        response = llm.generate(final_prompt)
        return self._parse_response_to_fields(response)

    def _extract_with_user_template(self, llm, document, chunks, schema, embedder, template):
        """User provided prompt template with {context} and {schema} placeholders."""
        import json as json_mod
        if schema and embedder:
            search_terms = " ".join(
                spec.get("description", name.replace("_", " "))
                for name, spec in schema.items()
            )
            relevant = self._retrieve_chunks(search_terms, chunks, embedder, top_k=self._chunk_top_k)
            context = "\n\n".join([c.text for c in relevant])
        else:
            context = "\n\n".join([c.text for c in chunks])

        schema_str = json_mod.dumps(schema, indent=2) if schema else ""
        final_prompt = template.replace("{context}", context).replace("{schema}", schema_str)
        response = llm.generate(final_prompt)
        return self._parse_response_to_fields(response, schema)

    def _extract_with_schema_rag(self, llm, document, chunks, schema, embedder):
        """For each field in schema, retrieve relevant chunks and extract."""
        all_fields = {}

        # Group fields by likely location to reduce LLM calls
        # Instead of 1 call per field, batch related fields together
        field_groups = self._group_fields(schema)

        for group_name, group_fields in field_groups.items():
            # Build search query from field descriptions
            search_terms = []
            for fname, fspec in group_fields.items():
                search_terms.append(fspec.get("description", fname.replace("_", " ")))
            search_query = " ".join(search_terms)

            # Retrieve relevant chunks
            relevant_chunks = self._retrieve_chunks(
                search_query, chunks, embedder, top_k=self._chunk_top_k
            )
            context = "\n\n".join([c.text for c in relevant_chunks])

            # Build prompt with only relevant context
            field_list = []
            for fname, fspec in group_fields.items():
                desc = fspec.get("description", fname.replace("_", " "))
                ftype = fspec.get("type", "string")
                field_list.append(f"  {fname} ({ftype}): {desc}")
            fields_text = "\n".join(field_list)

            prompt = (
                f"Extract these fields from the document context below.\n\n"
                f"Fields:\n{fields_text}\n\n"
                f"Context:\n{context}\n\n"
                f"Return JSON with field names as keys. "
                f"Numbers without currency symbols. Dates as YYYY-MM-DD. "
                f"If not found set null.\n"
                f"JSON:"
            )

            response = llm.generate(prompt)
            parsed = self._parse_json(response)

            for fname, value in parsed.items():
                if value is not None and str(value).lower() not in ("null", "none", "n/a", ""):
                    ftype = group_fields.get(fname, {}).get("type", "string")
                    typed_value = self._type_value(value, ftype)
                    # Find which chunk had this info
                    source_chunk = self._find_source_chunk(str(value), relevant_chunks)
                    all_fields[fname] = FieldResult(
                        value=typed_value, confidence=0.93,
                        raw_text=str(value), extraction_method="rag",
                        page=source_chunk.metadata.page_num if source_chunk else 0,
                    )

        return all_fields

    def _extract_auto_rag(self, llm, document, chunks, embedder):
        """Auto extraction using RAG — process all chunks, merge results."""
        all_fields = {}

        # For auto extraction, process chunks in batches
        batch_size = self._chunk_top_k
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            context = "\n\n".join([c.text for c in batch])

            prompt = (
                f"Extract all key information from this document section as key-value pairs.\n\n"
                f"Context:\n{context}\n\n"
                f"Return JSON. Numbers without currency symbols. Dates as YYYY-MM-DD.\n"
                f"JSON:"
            )

            response = llm.generate(prompt)
            parsed = self._parse_json(response)

            for key, value in parsed.items():
                if key in ("document_type", "doc_type", "type", "section"):
                    continue
                if value is not None and str(value).lower() not in ("null", "none", "n/a", ""):
                    # Don't overwrite existing fields (first mention wins)
                    if key not in all_fields:
                        all_fields[key] = FieldResult(
                            value=value, confidence=0.90,
                            raw_text=str(value), extraction_method="rag",
                        )

        return all_fields

    def _retrieve_chunks(self, query: str, chunks: list[DocqwiseChunk],
                          embedder, top_k: int = 5) -> list[DocqwiseChunk]:
        """Retrieve most relevant chunks for a query."""
        if embedder is None or not any(c.embedding is not None for c in chunks):
            # No embeddings — return first N chunks (positional)
            return chunks[:top_k]

        import numpy as np
        query_emb = embedder.embed(query)

        scored = []
        for chunk in chunks:
            if chunk.embedding is not None:
                similarity = float(np.dot(query_emb, chunk.embedding) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(chunk.embedding) + 1e-10
                ))
                scored.append((chunk, similarity))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, s in scored[:top_k]]

    def _group_fields(self, schema: dict) -> dict[str, dict]:
        """Group related fields to reduce LLM calls."""
        # Simple grouping: if few fields, keep as one group
        if len(schema) <= 6:
            return {"all": schema}

        # Group by likely document section
        groups = {
            "header": {},
            "financial": {},
            "details": {},
            "other": {},
        }
        financial_keywords = {"total", "amount", "price", "cost", "tax", "subtotal", "fee", "payment", "salary", "compensation"}
        header_keywords = {"name", "number", "date", "id", "reference", "vendor", "client", "company", "address"}

        for fname, fspec in schema.items():
            fname_lower = fname.lower()
            if any(kw in fname_lower for kw in financial_keywords):
                groups["financial"][fname] = fspec
            elif any(kw in fname_lower for kw in header_keywords):
                groups["header"][fname] = fspec
            elif fspec.get("type") in ("array", "object"):
                groups["details"][fname] = fspec
            else:
                groups["other"][fname] = fspec

        # Remove empty groups
        return {k: v for k, v in groups.items() if v}

    def _find_source_chunk(self, value: str, chunks: list[DocqwiseChunk]) -> Optional[DocqwiseChunk]:
        """Find which chunk contains the extracted value."""
        value_lower = value.lower()
        for chunk in chunks:
            if value_lower in chunk.text.lower():
                return chunk
        return None

    def _parse_response_to_fields(self, response: str, schema: dict = None) -> dict[str, FieldResult]:
        """Parse LLM response into FieldResult dict."""
        parsed = self._parse_json(response)
        fields = {}
        for key, value in parsed.items():
            if key in ("document_type", "doc_type", "type"):
                continue
            if value is None or str(value).lower() in ("null", "none", "n/a", ""):
                continue
            if schema and key in schema:
                ftype = schema[key].get("type", "string")
                value = self._type_value(value, ftype)
            fields[key] = FieldResult(
                value=value, confidence=0.90,
                raw_text=str(value), extraction_method="rag",
            )
        return fields

    def _parse_json(self, response: str) -> dict:
        """Parse JSON from LLM response."""
        text = response.strip()
        text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
        text = text.strip()

        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Find { ... } with brace counting
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
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        # Fix common issues
                        fixed = re.sub(r',\s*}', '}', candidate)
                        fixed = re.sub(r',\s*]', ']', fixed)
                        try:
                            return json.loads(fixed)
                        except json.JSONDecodeError:
                            pass
                    break

        # Extract key:value from text
        result = {}
        for line in text.split('\n'):
            line = line.strip().lstrip('- *')
            match = re.match(r'["\']?([a-zA-Z_]\w*)["\']?\s*[:=]\s*["\']?(.+?)["\']?\s*[,}]?\s*$', line)
            if match:
                key = match.group(1)
                val = match.group(2).strip().rstrip(',')
                result[key] = self._auto_type(val)

        return result

    def _type_value(self, value: Any, field_type: str) -> Any:
        if value is None:
            return None
        if field_type in ("number", "integer"):
            return self._to_number(value) or value
        if field_type == "boolean":
            return str(value).lower() in ("true", "yes", "1")
        if field_type == "array" and isinstance(value, str):
            try:
                return json.loads(value)
            except:
                return [v.strip() for v in value.split(",")]
        return value

    def _auto_type(self, value: str) -> Any:
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

    def _to_number(self, value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return value
        try:
            cleaned = re.sub(r'[,$\s]', '', str(value))
            cleaned = re.sub(r'(?:USD|INR|EUR|GBP|Rs\.?|₹)', '', cleaned).strip()
            if cleaned:
                return float(cleaned)
        except (ValueError, TypeError):
            pass
        return None
