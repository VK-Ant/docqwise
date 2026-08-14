"""Tests for pipeline and supporting components."""

def test_pipeline_dag():
    from docqwise.pipeline.pipeline import Pipeline
    pipe = Pipeline("test")
    pipe.add_node("a").add_node("b").add_node("c")
    pipe.connect("a", "b").connect("b", "c")
    assert pipe.validate() is True

def test_pipeline_cycle_detection():
    from docqwise.pipeline.pipeline import Pipeline
    pipe = Pipeline("cycle")
    pipe.add_node("a").add_node("b")
    pipe.connect("a", "b").connect("b", "a")
    assert pipe.validate() is False

def test_pipeline_topological_sort():
    from docqwise.pipeline.pipeline import Pipeline
    pipe = Pipeline("sort")
    pipe.add_node("read").add_node("extract").add_node("store")
    pipe.connect("read", "extract").connect("extract", "store")
    levels = pipe.topological_sort()
    assert levels[0] == ["read"]
    assert levels[-1] == ["store"]

def test_document_graph():
    from docqwise.graph.document_graph import DocumentGraph
    g = DocumentGraph()
    g.add_document("d1", {"type": "invoice"})
    g.add_entity("e1", "ORG", "Acme")
    g.add_edge("d1", "e1", "from")
    assert g.node_count == 2
    assert g.edge_count == 1
    assert len(g.communities()) > 0
    results = g.query("Acme")
    assert len(results) > 0

def test_classifier():
    from docqwise.classifier.zero_shot import ZeroShotClassifier
    c = ZeroShotClassifier()
    labels = c.classify("Invoice Number: INV-001. Total: $500. Payment due: Net 30.")
    assert labels[0]["label"] == "invoice"

def test_comparator():
    from docqwise.comparator.text_diff import TextComparator
    diff = TextComparator().compare("hello world", "hello there")
    assert "similarity" in diff
    assert diff["similarity"] < 1.0

def test_pii_detector():
    from docqwise.security.pii_detector import PIIDetector
    det = PIIDetector()
    matches = det.detect("email: test@example.com phone: +1-555-123-4567")
    assert len(matches) >= 2
    redacted = det.redact("email: test@example.com")
    assert "[REDACTED]" in redacted

def test_schema_detector():
    from docqwise.schema.detector import SchemaDetector
    s = SchemaDetector().detect(["name", "age"], [{"name": "A", "age": "25"}])
    assert len(s["fields"]) == 2

def test_quality_scorer():
    from docqwise.schema.quality import QualityScorer
    q = QualityScorer().score(["a"], [{"a": "x"}])
    assert q["overall"] > 0

def test_correction_store():
    import os, shutil, tempfile
    from docqwise.learning.correction import CorrectionStore
    path = os.path.join(tempfile.gettempdir(), "test_corr")
    if os.path.exists(path): shutil.rmtree(path, ignore_errors=True)
    store = CorrectionStore(path)
    store.save("d1", "f.pdf", {"tax": 100})
    assert store.count() == 1
    corrected = store.apply("invoice", {"tax": 0})
    assert corrected["tax"] == 100
    shutil.rmtree(path, ignore_errors=True)

def test_change_detector():
    import os, shutil, tempfile
    from docqwise.incremental.change_detector import ChangeDetector
    path = os.path.join(tempfile.gettempdir(), "test_change")
    if os.path.exists(path): shutil.rmtree(path, ignore_errors=True)
    cd = ChangeDetector(path)
    assert cd.has_changed("demo/sample_invoice.pdf") is True
    cd.mark_processed("demo/sample_invoice.pdf")
    assert cd.has_changed("demo/sample_invoice.pdf") is False
    shutil.rmtree(path, ignore_errors=True)
