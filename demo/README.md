# DocQWise Demos

## Run in order

| Demo | What it shows | Requirements |
|---|---|---|
| `01_quickstart.py` | All core features — no ML needed | `pip install docqwise` |
| `02_ollama.py` | AI extraction with local Ollama LLM | Ollama + `ollama pull nemotron-mini` |
| `03_huggingface.py` | AI extraction with HuggingFace on GPU | `pip install transformers torch bitsandbytes accelerate` |
| `04_rag.py` | RAG-based extraction (full pipeline) | `pip install sentence-transformers` + Ollama or HuggingFace |
| `05_graphrag.py` | GraphRAG + graph visualization | `pip install sentence-transformers` + Ollama |
| `local_demo.py` | Point at any folder — interactive Q&A | `pip install docqwise sentence-transformers` |

## Quick start

```bash
pip install docqwise
python demo/01_quickstart.py

# AI extraction
ollama pull nemotron-mini && python demo/02_ollama.py

# GraphRAG + visualization
python demo/05_graphrag.py
# Open docqwise_graph.html in your browser

# Point at your own files
python demo/local_demo.py "C:/MyDocuments"
```
