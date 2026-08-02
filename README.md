<p align="center">
  <img src="https://raw.githubusercontent.com/VK-Ant/docqwise/main/assets/hero.png" alt="DocQWise: Read, Extract, Retrieve" width="100%">
</p>

<p align="center">
  <strong>Document intelligence that adapts, accelerates, and scales.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/docqwise/"><img src="https://img.shields.io/badge/PyPI-docqwise-blue" alt="PyPI"></a>
  <a href="https://github.com/VK-Ant/docqwise/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://github.com/VK-Ant/docqwise/blob/main/notebooks/docqwise_getting_started.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</p>

---

## What is DocQWise?

DocQWise is a pluggable, AI-powered document intelligence engine. It reads any document format, extracts structured data using LLMs and RAG pipeline, and retrieves information with semantic search — locally, at scale, for zero per-page cost.

## Install

Choose your install based on what you need:

```bash
# Option 1: Core only (PDF reading, regex extraction, no ML)
pip install docqwise

# Option 2: With ML (RAG pipeline, embeddings, OCR — recommended)
pip install -r requirements-ml.txt

# Option 3: Everything (all features, all formats)
pip install -r requirements-full.txt
```

### LLM backend (pick one)

```bash
# Ollama — local, free, recommended
# Download from https://ollama.com then:
ollama pull nemotron-mini

# OR HuggingFace — local GPU
pip install transformers torch bitsandbytes accelerate

# OR OpenAI — cloud API
export OPENAI_API_KEY=your-key
```

## Quick Start

```python
from docqwise import Docqwise

dq = Docqwise()

# Ingest any document
dq.ingest("documents/")

# Extract fields with template
result = dq.extract_fields("invoice.pdf", template="invoice")
print(result.to_json())

# Ask questions about structured data
dq.ingest("sales.csv")
answer = dq.ask("What is the total amount?")
```

## Extraction Methods

```python
dq = Docqwise()

# RAG (default) — chunk → embed → retrieve → LLM extract
dq.extract_fields("doc.pdf", template="invoice")

# Direct LLM
dq.extract_fields("doc.pdf", template="invoice", method="llm")

# Vision (scanned docs, handwriting)
dq.extract_fields("scan.jpg", method="vision", model="gpt-4o")

# Regex (fast, no ML)
dq.extract_fields("doc.pdf", template="invoice", method="regex")
```

## LLM Backends

```python
# Ollama (local)
dq.extract_fields("doc.pdf", model="nemotron-mini")

# HuggingFace (local GPU)
from docqwise.llm.hf_llm import HuggingFaceLLM
llm = HuggingFaceLLM("Qwen/Qwen2.5-3B-Instruct", quantize="4bit")
dq.extract_fields("doc.pdf", llm=llm)

# OpenAI (cloud)
dq.extract_fields("doc.pdf", model="gpt-4o-mini")
```

## Templates

```python
dq.extract_fields("invoice.pdf", template="invoice")
dq.extract_fields("contract.pdf", template="contract")
dq.extract_fields("resume.pdf", template="resume")
dq.extract_fields("receipt.jpg", template="receipt")

# Custom schema
schema = {
    "vendor": {"type": "string", "description": "Company name"},
    "total": {"type": "number", "description": "Total amount"},
}
dq.extract_fields("doc.pdf", schema=schema)
```


## Custom Prompts

You design the prompts. We run the pipeline.

```python
dq = Docqwise()

# Default — docqwise handles the prompt
dq.extract_fields("doc.pdf", template="invoice")

# Your own prompt — full control
dq.extract_fields("doc.pdf", prompt="""
You are a medical record parser.
Extract patient name, diagnosis, and prescribed medications.
Return JSON only.

Document:
{context}

JSON:
""")

# Your prompt template with schema
dq.extract_fields("doc.pdf",
    schema={"patient": {"type": "string"}, "diagnosis": {"type": "string"}},
    prompt_template="""
Given this extraction schema:
{schema}

Parse this document:
{context}

Return JSON matching the schema exactly.
""")
```

## Self-Improving Corrections

```python
result = dq.extract_fields("invoice.pdf", template="invoice")
result.correct({"tax": 33300.00, "gst_number": "29AABCU9603R1ZM"})
# Next similar document → corrections applied automatically
```

## Structured Data Q&A

```python
dq.ingest("sales.xlsx")
dq.ask("What is the total amount?")         # exact SUM
dq.ask("Which vendor has highest sales?")    # GROUP BY + MAX
dq.ask("How many invoices are overdue?")     # COUNT + WHERE
```

## All Features

```python
dq = Docqwise()

# Ingestion
dq.ingest("file.pdf")                    # single file
dq.ingest("documents/")                  # folder (all formats)
dq.ingest("data.csv")                    # structured data

# Extraction
dq.extract_fields("doc.pdf")             # field extraction
dq.extract_tables("doc.pdf")             # table extraction
dq.extract_entities("doc.pdf")           # entity extraction
dq.extract_images("doc.pdf")             # image extraction
dq.extract_text("doc.pdf")               # text extraction
dq.auto_extract("doc.pdf")               # auto-detect + extract

# Intelligence
dq.retrieve("query", top_k=5)            # semantic search
dq.ask("question")                       # Q&A
dq.classify("doc.pdf")                   # classification
dq.compare("v1.pdf", "v2.pdf")           # comparison
dq.detect_schema("data.csv")             # schema detection
dq.detect_pii("doc.pdf")                 # PII detection
```

## Demos

Run in order:

| Demo | What | Install |
|---|---|---|
| `python demo/01_quickstart.py` | All core features | `pip install docqwise` |
| `python demo/02_ollama.py` | AI extraction with Ollama | `ollama pull nemotron-mini` |
| `python demo/03_huggingface.py` | AI extraction on GPU | `pip install transformers torch bitsandbytes accelerate` |
| `python demo/04_rag.py` | Full RAG pipeline | `pip install sentence-transformers` |

## Notebook

```bash
pip install jupyter
jupyter notebook notebooks/docqwise_getting_started.ipynb
```

## Testing

```bash
pip install pytest
pytest -v
```

## Docker

```bash
docker compose up --build
```

## Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/VK-Ant/docqwise/main/assets/arc.png" alt="arc" width="100%">
</p>


```
engine.py (stable — never changes)
    └── factory.py (all component selection)
            ├── ExtractorFactory  → rag | llm | vision | regex
            ├── LLMFactory        → ollama | huggingface | openai
            ├── EmbedderFactory   → sentence-transformers | any
            ├── StoreFactory      → sqlite | qdrant | faiss | any
            ├── ChunkerFactory    → structure | fixed | sentence
            └── TemplateFactory   → invoice | contract | resume | receipt
```

## Ecosystem

| Library | Tagline | Domain |
|---|---|---|
| [SightRAG](https://github.com/VK-Ant/SightRAG) | See. Search. Retrieve. | Visual intelligence |
| [sonarwise](https://github.com/VK-Ant/sonarwise) | Hear. Search. Retrieve. | Audio intelligence |
| [docqwise](https://github.com/VK-Ant/docqwise) | **Read. Extract. Retrieve.** | **Document intelligence** |
| [adaptive-intelligence](https://github.com/VK-Ant/adaptive-intelligence) | Learn. Remember. Adapt. | Orchestration |
| [llmevalkit](https://github.com/VK-Ant/llmevalkit) | Evaluate. Score. Improve. | Evaluation |

## License

Apache License 2.0

## Author

**Venkatkumar Rajan**