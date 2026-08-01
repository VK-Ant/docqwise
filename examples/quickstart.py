"""
DocQWise Quick Start
====================

Basic usage examples for docqwise.
"""

from docqwise import Docqwise

# ── Basic setup ──
dq = Docqwise()
print(dq)
print(dq.metrics())

# ── Ingest documents ──
# dq.ingest("documents/")              # folder (auto-detect all formats)
# dq.ingest("report.pdf")              # single PDF
# dq.ingest("data.xlsx")               # Excel spreadsheet
# dq.ingest("postgresql://host/db", tables=["invoices"])  # database

# ── Extract fields ──
# fields = dq.extract_fields("invoice.pdf", template="invoice")
# print(fields.to_dict())

# ── Extract tables ──
# tables = dq.extract_tables("report.pdf")
# df = tables[0].to_dataframe()

# ── Semantic search ──
# results = dq.retrieve("payment terms", top_k=5)

# ── Natural language Q&A ──
# answer = dq.ask("What is the total amount?")

# ── Self-improving corrections ──
# result = dq.extract_fields("invoice.pdf", template="invoice")
# result.correct({"tax": 1402.00})
# Next similar invoice will extract correctly

# ── Production setup ──
# dq = Docqwise(
#     strategy="hybrid",
#     workers=8,
#     gpu_workers=2,
#     vector_store="qdrant",
#     search_mode="hybrid",
# )
