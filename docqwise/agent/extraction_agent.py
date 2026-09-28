"""Agentic document extraction — multi-pass, self-correcting extraction.

The ExtractionAgent runs an autonomous loop:
    1. Initial extraction pass (RAG or LLM)
    2. Validate extracted fields against schema rules
    3. Re-extract failed fields with focused context
    4. Cross-check between extraction methods
    5. Confidence scoring and final merge

Usage:
    from docqwise.agent import ExtractionAgent

    agent = ExtractionAgent(llm=my_llm)
    result = agent.extract("invoice.pdf", schema=my_schema)
    print(result.fields)
    print(agent.trace)  # full audit trail
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from docqwise.core.document import DocqwiseDocument
from docqwise.core.field import FieldResult, ExtractionResult

logger = logging.getLogger("docqwise")


# ══════════════════════════════════════════
# VALIDATION RULES
# ══════════════════════════════════════════

@dataclass
class ValidationRule:
    """A rule for validating an extracted field."""
    field_name: str
    rule_type: str  # required, type, regex, range, choices, custom
    params: dict = field(default_factory=dict)
    message: str = ""

    def check(self, value: Any) -> tuple[bool, str]:
        """Returns (passed, reason)."""
        if self.rule_type == "required":
            ok = value is not None and str(value).strip() != ""
            return ok, "" if ok else f"{self.field_name} is required but empty"

        if self.rule_type == "type":
            expected = self.params.get("expected", "str")
            try:
                if expected == "int":
                    int(str(value).replace(",", ""))
                elif expected == "float":
                    float(str(value).replace(",", ""))
                elif expected == "date":
                    # Rough date check
                    if not re.search(r'\d{1,4}[/-]\d{1,2}[/-]\d{1,4}', str(value)):
                        return False, f"{self.field_name} does not look like a date"
                return True, ""
            except (ValueError, TypeError):
                return False, f"{self.field_name} should be {expected}, got: {value}"

        if self.rule_type == "regex":
            pattern = self.params.get("pattern", "")
            ok = bool(re.search(pattern, str(value)))
            return ok, "" if ok else f"{self.field_name} does not match pattern {pattern}"

        if self.rule_type == "range":
            try:
                num = float(str(value).replace(",", "").replace("$", "").replace("€", "").replace("£", "").replace("₹", ""))
                lo = self.params.get("min")
                hi = self.params.get("max")
                if lo is not None and num < lo:
                    return False, f"{self.field_name} = {num} is below minimum {lo}"
                if hi is not None and num > hi:
                    return False, f"{self.field_name} = {num} exceeds maximum {hi}"
                return True, ""
            except (ValueError, TypeError):
                return False, f"{self.field_name} is not numeric for range check"

        if self.rule_type == "choices":
            allowed = self.params.get("values", [])
            ok = str(value).strip().lower() in [str(v).lower() for v in allowed]
            return ok, "" if ok else f"{self.field_name} = '{value}' not in {allowed}"

        return True, ""


def rules_from_schema(schema: dict) -> list[ValidationRule]:
    """Derive validation rules from a field schema."""
    rules = []
    for field_name, spec in schema.items():
        if isinstance(spec, str):
            # Simple schema: {"invoice_number": "string"}
            rules.append(ValidationRule(field_name, "required"))
            if spec in ("int", "integer", "number"):
                rules.append(ValidationRule(field_name, "type", {"expected": "float"}))
            elif spec in ("date",):
                rules.append(ValidationRule(field_name, "type", {"expected": "date"}))
        elif isinstance(spec, dict):
            if spec.get("required", True):
                rules.append(ValidationRule(field_name, "required"))
            if "type" in spec:
                rules.append(ValidationRule(field_name, "type",
                                           {"expected": spec["type"]}))
            if "pattern" in spec:
                rules.append(ValidationRule(field_name, "regex",
                                           {"pattern": spec["pattern"]}))
            if "min" in spec or "max" in spec:
                rules.append(ValidationRule(field_name, "range", {
                    "min": spec.get("min"), "max": spec.get("max"),
                }))
            if "choices" in spec:
                rules.append(ValidationRule(field_name, "choices",
                                           {"values": spec["choices"]}))
    return rules


# ══════════════════════════════════════════
# AGENT TRACE
# ══════════════════════════════════════════

@dataclass
class AgentStep:
    """One step in the agent's execution trace."""
    step: int
    action: str  # extract, validate, retry, cross_check, merge
    fields_affected: list[str] = field(default_factory=list)
    details: str = ""
    duration_ms: float = 0.0
    success: bool = True


@dataclass
class AgentTrace:
    """Full audit trail of the agent's extraction run."""
    steps: list[AgentStep] = field(default_factory=list)
    total_passes: int = 0
    retried_fields: list[str] = field(default_factory=list)
    cross_checked: bool = False
    final_confidence: float = 0.0
    total_duration_ms: float = 0.0

    def add(self, action: str, fields: list[str] = None,
            details: str = "", duration_ms: float = 0.0, success: bool = True):
        self.steps.append(AgentStep(
            step=len(self.steps) + 1,
            action=action,
            fields_affected=fields or [],
            details=details,
            duration_ms=duration_ms,
            success=success,
        ))

    def summary(self) -> str:
        lines = [f"Agent trace: {self.total_passes} passes, "
                 f"{len(self.retried_fields)} retries, "
                 f"confidence={self.final_confidence:.2f}"]
        for s in self.steps:
            status = "✓" if s.success else "✗"
            lines.append(f"  {status} Step {s.step}: {s.action} "
                         f"({s.duration_ms:.0f}ms) {s.details}")
        return "\n".join(lines)


# ══════════════════════════════════════════
# EXTRACTION AGENT
# ══════════════════════════════════════════

class ExtractionAgent:
    """Multi-pass, self-correcting document extraction agent.

    The agent autonomously:
    1. Runs initial extraction (RAG-based by default)
    2. Validates results against schema rules
    3. Retries failed fields with focused prompts
    4. Optionally cross-checks with a second extraction method
    5. Merges results with confidence weighting

    Args:
        llm: LLM instance (auto-detected if None)
        embedder: Embedder instance (auto-detected if None)
        max_retries: Maximum retry passes for failed fields (default 2)
        cross_check: Whether to use a second method for verification (default True)
        confidence_threshold: Minimum confidence to accept a field (default 0.6)
        validation_rules: Custom validation rules (auto-derived from schema if None)
        model: Ollama model name override

    Usage:
        agent = ExtractionAgent(max_retries=3, cross_check=True)
        result = agent.extract("invoice.pdf", template="invoice")
        print(agent.trace.summary())
    """

    def __init__(
        self,
        llm=None,
        embedder=None,
        max_retries: int = 2,
        cross_check: bool = True,
        confidence_threshold: float = 0.6,
        validation_rules: list[ValidationRule] = None,
        model: str = None,
    ):
        self._llm = llm
        self._embedder = embedder
        self._max_retries = max_retries
        self._cross_check = cross_check
        self._confidence_threshold = confidence_threshold
        self._custom_rules = validation_rules
        self._model = model
        self.trace = AgentTrace()

    def _get_reader(self):
        from docqwise.readers.auto_reader import AutoReader
        return AutoReader()

    def _get_rag_extractor(self):
        from docqwise.extractors.rag_extractor import RAGExtractor
        return RAGExtractor(
            llm=self._llm, embedder=self._embedder,
            model=self._model,
        )

    def _get_llm_extractor(self):
        from docqwise.extractors.llm_extractor import LLMFieldExtractor
        return LLMFieldExtractor(llm=self._llm, model=self._model)

    def _get_regex_extractor(self):
        from docqwise.extractors.field_extractor import RegexFieldExtractor
        return RegexFieldExtractor()

    def extract(
        self,
        source: str,
        schema: dict = None,
        template: str = None,
        prompt: str = None,
        prompt_template: str = None,
        system_prompt: str = None,
        **kwargs,
    ) -> ExtractionResult:
        """Run multi-pass agentic extraction.

        Args:
            source: Document file path
            schema: Field definitions {field_name: type_or_spec}
            template: Pre-built template name (invoice, contract, etc.)
            prompt: Custom extraction prompt
            prompt_template: Prompt with {text} and {schema} placeholders
            system_prompt: System prompt for LLM

        Returns:
            ExtractionResult with all extracted fields
        """
        start = time.time()
        self.trace = AgentTrace()

        # Read document
        reader = self._get_reader()
        doc = reader.read(source)

        # Resolve schema from template
        target_schema = schema
        if template and not target_schema:
            from docqwise.factory import TemplateFactory
            tmpl = TemplateFactory.get(template)
            if tmpl:
                target_schema = tmpl.SCHEMA

        if not target_schema:
            target_schema = self._auto_detect_schema(doc)

        # Build validation rules
        rules = self._custom_rules or rules_from_schema(target_schema)

        # ── Pass 1: Initial extraction (RAG) ──
        t1 = time.time()
        try:
            rag = self._get_rag_extractor()
            result = rag.extract_fields(
                doc, schema=target_schema,
                prompt=prompt, prompt_template=prompt_template,
            )
            self.trace.add("extract", list(result.fields.keys()),
                           f"RAG extraction: {len(result.fields)} fields",
                           (time.time() - t1) * 1000)
            self.trace.total_passes = 1
        except Exception as e:
            logger.warning(f"RAG extraction failed: {e}, falling back to regex")
            self.trace.add("extract", [], f"RAG failed: {e}", (time.time() - t1) * 1000, False)
            result = self._fallback_regex(doc, target_schema)
            self.trace.total_passes = 1

        # ── Pass 2: Validate ──
        t2 = time.time()
        failed_fields = self._validate(result, rules)
        self.trace.add("validate",
                       [f for f, _ in failed_fields],
                       f"{len(failed_fields)} fields failed validation",
                       (time.time() - t2) * 1000,
                       len(failed_fields) == 0)

        # ── Pass 3+: Retry failed fields ──
        for attempt in range(self._max_retries):
            if not failed_fields:
                break

            t3 = time.time()
            retry_schema = {f: target_schema.get(f, "string") for f, _ in failed_fields}
            retry_fields = [f for f, _ in failed_fields]

            retried = self._retry_extraction(
                doc, retry_schema, failed_fields,
                prompt=prompt, system_prompt=system_prompt,
            )

            # Merge retried fields into result
            merged = 0
            for fname, field_result in retried.items():
                if field_result.value and str(field_result.value).strip():
                    result.fields[fname] = field_result
                    merged += 1

            self.trace.add("retry",
                           retry_fields,
                           f"Attempt {attempt + 1}: retried {len(retry_fields)}, "
                           f"fixed {merged}",
                           (time.time() - t3) * 1000,
                           merged > 0)
            self.trace.total_passes += 1
            self.trace.retried_fields.extend(retry_fields)

            # Re-validate
            failed_fields = self._validate(result, rules)

        # ── Pass 4: Cross-check (optional) ──
        if self._cross_check:
            t4 = time.time()
            self._do_cross_check(doc, result, target_schema)
            self.trace.cross_checked = True
            self.trace.add("cross_check", list(result.fields.keys()),
                           "Cross-checked with regex extractor",
                           (time.time() - t4) * 1000)
            self.trace.total_passes += 1

        # ── Final: Compute confidence ──
        t5 = time.time()
        self._compute_final_confidence(result, target_schema)
        self.trace.final_confidence = result.confidence
        self.trace.add("merge", list(result.fields.keys()),
                       f"Final confidence: {result.confidence:.2f}",
                       (time.time() - t5) * 1000)

        elapsed = (time.time() - start) * 1000
        self.trace.total_duration_ms = elapsed
        result.processing_time_ms = elapsed
        result.strategy_used = "agentic"

        logger.info(f"Agentic extraction done: {len(result.fields)} fields, "
                    f"{self.trace.total_passes} passes, "
                    f"confidence={result.confidence:.2f}, "
                    f"{elapsed:.0f}ms")

        return result

    def _auto_detect_schema(self, doc: DocqwiseDocument) -> dict:
        """Try to detect a schema from document content when none provided."""
        # Use document profiler / classifier to guess type
        try:
            from docqwise.classifier.zero_shot import ZeroShotClassifier
            from docqwise.factory import TemplateFactory
            labels = ZeroShotClassifier().classify(doc.text[:2000])
            if labels:
                top = labels[0]["label"]
                tmpl = TemplateFactory.get(top)
                if tmpl:
                    return tmpl.SCHEMA
        except Exception:
            pass

        # Fallback: generic extraction schema
        return {
            "document_type": "string",
            "date": "date",
            "total_amount": "number",
            "parties": "string",
            "reference_number": "string",
        }

    def _validate(self, result: ExtractionResult,
                  rules: list[ValidationRule]) -> list[tuple[str, str]]:
        """Validate extracted fields against rules. Returns [(field, reason)]."""
        failures = []
        for rule in rules:
            field_result = result.fields.get(rule.field_name)
            value = field_result.value if field_result else None
            passed, reason = rule.check(value)
            if not passed:
                failures.append((rule.field_name, reason))
        return failures

    def _retry_extraction(self, doc: DocqwiseDocument,
                          retry_schema: dict,
                          failures: list[tuple[str, str]],
                          prompt: str = None,
                          system_prompt: str = None) -> dict[str, FieldResult]:
        """Retry extraction for specific failed fields with focused prompts."""
        results = {}

        # Build a focused prompt that tells the LLM exactly what failed
        failure_context = "\n".join(
            f"- {fname}: {reason}" for fname, reason in failures
        )

        focused_prompt = (
            f"Extract ONLY these fields from the document. "
            f"Previous extraction failed for these fields:\n{failure_context}\n\n"
            f"Look carefully in the document for these values. "
            f"Fields to extract: {json.dumps(list(retry_schema.keys()))}\n\n"
            f"Document text:\n{{text}}\n\n"
            f"Return valid JSON with these fields: {json.dumps(list(retry_schema.keys()))}"
        )

        try:
            # Use LLM extractor for retry (direct, no RAG chunking overhead)
            extractor = self._get_llm_extractor()
            retry_result = extractor.extract_fields(
                doc, schema=retry_schema,
                prompt_template=focused_prompt,
            )
            for fname, field_val in retry_result.fields.items():
                if fname in retry_schema:
                    field_val.extraction_method = "agentic_retry"
                    results[fname] = field_val
        except Exception as e:
            logger.warning(f"Retry extraction failed: {e}")

            # Fallback: try regex for the failed fields
            try:
                regex = self._get_regex_extractor()
                regex_result = regex.extract_fields(doc, schema=retry_schema)
                for fname, field_val in regex_result.fields.items():
                    if fname in retry_schema and fname not in results:
                        field_val.extraction_method = "agentic_retry_regex"
                        results[fname] = field_val
            except Exception:
                pass

        return results

    def _do_cross_check(self, doc: DocqwiseDocument,
                        result: ExtractionResult,
                        schema: dict) -> None:
        """Cross-check results with regex extractor for verification."""
        try:
            regex = self._get_regex_extractor()
            regex_result = regex.extract_fields(doc, schema=schema)

            for fname, regex_field in regex_result.fields.items():
                if fname not in result.fields:
                    # New field found by regex — add it
                    regex_field.extraction_method = "cross_check_regex"
                    regex_field.confidence = max(0.5, regex_field.confidence - 0.1)
                    result.fields[fname] = regex_field
                elif result.fields[fname].value and regex_field.value:
                    rag_val = str(result.fields[fname].value).strip()
                    regex_val = str(regex_field.value).strip()

                    if rag_val.lower() == regex_val.lower():
                        # Agreement → boost confidence
                        result.fields[fname].confidence = min(
                            1.0, result.fields[fname].confidence + 0.1
                        )
                    elif not rag_val:
                        # RAG empty, regex has value → use regex
                        regex_field.extraction_method = "cross_check_regex"
                        result.fields[fname] = regex_field
                    # If they disagree and RAG has a value, keep RAG (higher trust)
        except Exception as e:
            logger.debug(f"Cross-check skipped: {e}")

    def _fallback_regex(self, doc: DocqwiseDocument,
                        schema: dict) -> ExtractionResult:
        """Last-resort regex extraction when everything else fails."""
        try:
            regex = self._get_regex_extractor()
            return regex.extract_fields(doc, schema=schema)
        except Exception:
            return ExtractionResult(
                doc_id=doc.doc_id,
                source_path=getattr(doc, "source_path", ""),
                fields={},
                confidence=0.0,
                strategy_used="agentic_fallback",
            )

    def _compute_final_confidence(self, result: ExtractionResult,
                                   schema: dict) -> None:
        """Compute overall confidence score from individual field scores."""
        if not result.fields:
            result.confidence = 0.0
            return

        total_expected = len(schema)
        total_extracted = len(result.fields)

        # Coverage score
        coverage = total_extracted / max(total_expected, 1)

        # Average field confidence
        confidences = [f.confidence for f in result.fields.values()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        # Weighted: 60% avg confidence, 40% coverage
        result.confidence = round(0.6 * avg_confidence + 0.4 * coverage, 3)

        # Add warnings for low-confidence fields
        for fname, fval in result.fields.items():
            if fval.confidence < self._confidence_threshold:
                result.warnings.append(
                    f"Low confidence for '{fname}': {fval.confidence:.2f}"
                )
