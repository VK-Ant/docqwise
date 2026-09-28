"""Tests for agentic extraction module."""

from docqwise.agent.extraction_agent import (
    ExtractionAgent,
    ValidationRule,
    AgentTrace,
    AgentStep,
    rules_from_schema,
)


def test_validation_rule_creation():
    rule = ValidationRule("total_amount", "required")
    assert rule.field_name == "total_amount"
    assert rule.rule_type == "required"
    assert rule.params == {}


def test_validation_rule_with_params():
    rule = ValidationRule("total_amount", "range", {"min": 0, "max": 100000})
    assert rule.params["min"] == 0
    assert rule.params["max"] == 100000


def test_validation_rule_check_required_pass():
    rule = ValidationRule("name", "required")
    passed, reason = rule.check("Acme Corp")
    assert passed is True


def test_validation_rule_check_required_fail():
    rule = ValidationRule("name", "required")
    passed, reason = rule.check(None)
    assert passed is False
    assert "required" in reason


def test_validation_rule_check_range_pass():
    rule = ValidationRule("amount", "range", {"min": 0, "max": 1000})
    passed, reason = rule.check("500")
    assert passed is True


def test_validation_rule_check_range_fail():
    rule = ValidationRule("amount", "range", {"min": 0, "max": 1000})
    passed, reason = rule.check("5000")
    assert passed is False
    assert "exceeds" in reason


def test_validation_rule_check_regex_pass():
    rule = ValidationRule("email", "regex", {"pattern": r"[\w.]+@[\w.]+\.\w+"})
    passed, _ = rule.check("test@example.com")
    assert passed is True


def test_validation_rule_check_regex_fail():
    rule = ValidationRule("email", "regex", {"pattern": r"[\w.]+@[\w.]+\.\w+"})
    passed, _ = rule.check("not-an-email")
    assert passed is False


def test_validation_rule_check_choices_pass():
    rule = ValidationRule("status", "choices", {"values": ["paid", "unpaid", "pending"]})
    passed, _ = rule.check("paid")
    assert passed is True


def test_validation_rule_check_choices_fail():
    rule = ValidationRule("status", "choices", {"values": ["paid", "unpaid"]})
    passed, _ = rule.check("cancelled")
    assert passed is False


def test_rules_from_schema():
    schema = {
        "invoice_number": "string",
        "total_amount": "number",
        "date": "date",
        "vendor_name": "string",
    }
    rules = rules_from_schema(schema)
    assert len(rules) >= 4
    field_names = [r.field_name for r in rules]
    assert "invoice_number" in field_names
    assert "total_amount" in field_names


def test_rules_from_schema_with_type_rules():
    schema = {
        "total_amount": "number",
        "date": "date",
    }
    rules = rules_from_schema(schema)
    type_rules = [r for r in rules if r.rule_type == "type"]
    assert len(type_rules) >= 1


def test_agent_trace():
    trace = AgentTrace()
    trace.add("extract", ["field1", "field2"], "RAG extraction", 50.0)
    trace.add("validate", ["field1"], "1 field failed", 10.0, success=False)
    assert len(trace.steps) == 2
    assert trace.steps[0].action == "extract"
    assert trace.steps[1].action == "validate"
    assert trace.steps[1].success is False


def test_agent_trace_summary():
    trace = AgentTrace()
    trace.add("extract", ["field1"], "RAG extraction", 50.0)
    trace.total_passes = 1
    trace.final_confidence = 0.85
    summary = trace.summary()
    assert "extract" in summary
    assert "1 passes" in summary
    assert isinstance(summary, str)


def test_agent_step():
    step = AgentStep(step=1, action="extract", fields_affected=["a", "b"],
                     details="test", duration_ms=100.0)
    assert step.step == 1
    assert step.action == "extract"
    assert step.fields_affected == ["a", "b"]
    assert step.duration_ms == 100.0
    assert step.success is True


def test_agent_creation():
    agent = ExtractionAgent(max_retries=3, cross_check=False)
    assert agent._max_retries == 3
    assert agent._cross_check is False
    assert agent._confidence_threshold == 0.6


def test_agent_default_creation():
    agent = ExtractionAgent()
    assert agent._max_retries == 2
    assert agent._cross_check is True
    assert agent._confidence_threshold == 0.6


def test_agent_custom_validation_rules():
    rules = [
        ValidationRule("total", "required"),
        ValidationRule("total", "range", {"min": 0, "max": 999999}),
    ]
    agent = ExtractionAgent(validation_rules=rules)
    assert len(agent._custom_rules) == 2


def test_agent_trace_reset_on_extract():
    """Verify trace is fresh AgentTrace on each agent instance."""
    agent = ExtractionAgent()
    assert isinstance(agent.trace, AgentTrace)
    assert len(agent.trace.steps) == 0
