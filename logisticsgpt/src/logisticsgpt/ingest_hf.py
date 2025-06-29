"""Download the Cainiao-AI LaDe(-D) dataset from HuggingFace and ingest it into the existing Qdrant
vector store so that LogisticsGPT can RAG over real last-mile logistics data.

Run:
    python -m logisticsgpt.ingest_hf --dataset Cainiao-AI/LaDe-D --split default --max_rows 100000
"""

import argparse, json
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from llama_index import SimpleNodeParser, VectorStoreIndex, StorageContext
from llama_index.embeddings import HuggingFaceEmbedding
from qdrant_client import QdrantClient

from .config import VECTOR_DB_PATH

CACHE_CSV = Path("lade_sample.csv")


def _download_to_csv(dataset_name: str, split: str, max_rows: int | None) -> pd.DataFrame:
    """Download the dataset from HF and materialize (optionally sampled) rows to a UTF-8-SIG CSV."""
    print(f"⬇️  Loading '{dataset_name}' split='{split}' via 🤗 datasets …")
    ds = load_dataset(dataset_name, split=split)
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))
    df = ds.to_pandas()
    print(f"✅ Loaded {len(df):,} rows; caching ⇒ {CACHE_CSV.relative_to(Path.cwd())}")
    df.to_csv(CACHE_CSV, index=False, encoding="utf-8-sig")
    return df


def _build_nodes(df: pd.DataFrame):
    """Convert each row to a JSON blob, preserving key fields for metadata filtering."""
    docs = [
        {
            "text": json.dumps(row.to_dict(), ensure_ascii=False),
            "metadata": {
                "order_id": row.get("order_id"),
                "region_id": row.get("region_id"),
            },
        }
        for _, row in df.iterrows()
    ]
    parser = SimpleNodeParser.from_defaults()
    return parser.get_nodes_from_documents(docs)


def ingest(dataset_name: str, split: str, max_rows: int | None):
    df = _download_to_csv(dataset_name, split, max_rows)
    nodes = _build_nodes(df)

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    client = QdrantClient(path=VECTOR_DB_PATH)

    VectorStoreIndex.from_documents(
        nodes,
        storage_context=StorageContext.from_defaults(vector_store=client),
        embed_model=embed_model,
    )
    print(
        f"🎉 Ingested {len(nodes):,} documents from '{dataset_name}' into Qdrant vector store at {VECTOR_DB_PATH}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingest LaDe dataset into Qdrant for LogisticsGPT RAG")
    ap.add_argument("--dataset", default="Cainiao-AI/LaDe-D", help="HF dataset name (default: %(default)s)")
    ap.add_argument("--split", default="default", help="Dataset split (default: %(default)s)")
    ap.add_argument(
        "--max_rows",
        type=int,
        default=100_000,
        help="Optional cap on number of rows to ingest (useful for local demos)",
    )
    args = ap.parse_args()

    ingest(args.dataset, args.split, args.max_rows)
