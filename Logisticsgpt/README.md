# LogisticsGPT – Full Agentic RAG Stack

This repository demonstrates how to:

* **Ingest** Cainiao LaDe‑D dataset into Qdrant (`ingest_hf.py`)
* **Load** a quantised SOTA LLM (DeepSeek/Grok/Claude via GGUF) with llama.cpp
* **Retrieve‑Augment** queries using the vector index (`ask_rag`)
* **Run agentic SQL commands** in natural language (`sql_query` / `sql_write`)
* **Generate charts** from DataFrame (`plot_dataframe`)

> Generated on 2025-06-29

## Quick start

```bash
pip install -r requirements.txt

# Ingest 100k rows for demo
python -m logisticsgpt.ingest_hf --max_rows 100000

# Ask a question
python -m logisticsgpt.demo_app --task rag --arg "How many parcels were delivered in region 2 in May 2024?"

# Read inventory via NL → SQL
python -m logisticsgpt.demo_app --task read --arg "Show top 10 SKUs with lowest stock"

# Write (guard‑railed) SQL
python -m logisticsgpt.demo_app --task write --arg "UPDATE inventory SET stock_qty = 42 WHERE sku = 'ABC123';"
```

See `src/logisticsgpt/sql_agent.py` for security guard rails (whitelists, destructive op blocking).
