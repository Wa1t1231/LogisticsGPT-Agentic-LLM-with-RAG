"""Centralised configuration for LogisticsGPT."""

from pathlib import Path
import os

# Vector DB (Qdrant) local path
VECTOR_DB_PATH: Path = Path(os.getenv("VECTOR_DB_PATH", "vector_db"))

# PostgreSQL / SQLite connection URI for inventory tables
# e.g. "postgresql+psycopg://user:password@localhost:5432/inventory"
SQL_DATABASE_URI: str = os.getenv("SQL_DATABASE_URI", "sqlite:///inventory.db")

# Allowed tables & columns for UPDATE / INSERT to mitigate destructive writes
ALLOWED_WRITE_TABLES = {
    "inventory": {"sku", "stock_qty", "location"},
}
