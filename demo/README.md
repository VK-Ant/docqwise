# DocQWise Demos

## Run in order

| Demo | What it shows | Requirements |
|---|---|---|
| `01_quickstart.py` | All core features — no ML needed | `pip install docqwise` |
| `02_ollama.py` | AI extraction with local Ollama LLM | Ollama + `ollama pull nemotron-mini` |
| `03_huggingface.py` | AI extraction with HuggingFace on GPU | `pip install transformers torch bitsandbytes accelerate` |
| `04_rag.py` | RAG-based extraction (full pipeline) | `pip install sentence-transformers` + Ollama or HuggingFace |

## Quick start

```bash
# Step 1: Install
pip install docqwise

# Step 2: Run quickstart (works immediately, no ML needed)
python demo/01_quickstart.py

# Step 3: Try AI extraction (pick one)
ollama pull nemotron-mini && python demo/02_ollama.py
# or
pip install transformers torch && python demo/03_huggingface.py

# Step 4: Try RAG extraction
pip install sentence-transformers && python demo/04_rag.py
```

## Sample documents included

| File | Type | Description |
|---|---|---|
| `sample_invoice.pdf` | Invoice | Acme Corp → TechStart, INR 2,18,300 |
| `sample_contract.pdf` | Contract | Service agreement, 12 months, liability cap |
| `sample_sales.csv` | CSV | 10 sales records with vendor, amount, status |
| `sample_data.json` | JSON | Company departments with headcount and budget |
