# LogisticsGPT – LaDe Ingestion

Ingest the **Cainiao-AI/LaDe-D** logistics dataset into a Qdrant vector store
for Retrieval‑Augmented Generation (RAG) with LogisticsGPT.

## Quick start

```bash
pip install -r requirements.txt
python -m logisticsgpt.ingest_hf --dataset Cainiao-AI/LaDe-D --split default --max_rows 100000
```

## Technical notes

* Streaming download to keep RAM usage low  
* Row‑level JSON stored as text for lossless RAG  
* 128‑dim `BAAI/bge-small-en-v1.5` embeddings  
* Qdrant IVF‑PQ collection created automatically  

Generated on 2025-06-29.
