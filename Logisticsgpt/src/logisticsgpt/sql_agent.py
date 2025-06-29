"""Natural language ↔ SQL agent with read & controlled write abilities."""

from __future__ import annotations
import logging
import re
from typing import List, Any

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from langchain.sql_database import SQLDatabase
from langchain.agents import create_sql_agent
from langchain_openai import ChatOpenAI

from .config import SQL_DATABASE_URI, ALLOWED_WRITE_TABLES

LOGGER = logging.getLogger(__name__)
_engine = create_engine(SQL_DATABASE_URI, future=True)
_db = SQLDatabase(_engine)

# ---------------- Guard rails helpers ----------------
_dangerous_regex = re.compile(
    r"\\b(drop|alter|truncate|delete)\\b", re.IGNORECASE
)


def _sanitize_sql(stmt: str, allow_write: bool) -> None:
    if _dangerous_regex.search(stmt):
        raise ValueError("SQL contains potentially destructive operation.")
    if not allow_write and re.search(r"\\b(update|insert)\\b", stmt, re.I):
        raise ValueError("Write operations are disabled in read‑only mode.")


def _check_allowed_columns(stmt: str) -> None:
    """Very simple whitelist parser for UPDATE/INSERT."""
    for table, cols in ALLOWED_WRITE_TABLES.items():
        if re.search(fr"\\b{table}\\b", stmt, re.I):
            for match in re.finditer(r"SET\\s+([^;]+)", stmt, re.I):
                update_cols = {c.strip().split("=")[0] for c in match.group(1).split(',')}
                disallowed = update_cols - cols
                if disallowed:
                    raise ValueError(f"Columns not allowed: {disallowed}")
            break


# ---------------- Public API ----------------
def sql_query(nl_query: str) -> List[Any]:
    """Natural language read‑only query → SQL → result list."""
    # LangChain agent with READ ONLY
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    agent = create_sql_agent(llm, _db, agent_type="openai-tools", verbose=False)

    sql_stmt = agent.run(nl_query)
    _sanitize_sql(sql_stmt, allow_write=False)

    with _engine.connect() as conn:
        result = conn.execute(text(sql_stmt)).fetchall()
    return result


def sql_write(sql_stmt: str) -> str:
    """Execute whitelisted UPDATE/INSERT statements."""
    _sanitize_sql(sql_stmt, allow_write=True)
    _check_allowed_columns(sql_stmt)

    try:
        with _engine.begin() as conn:
            conn.execute(text(sql_stmt))
    except SQLAlchemyError as ex:
        LOGGER.exception("Write failed: %s", ex)
        raise
    return "Write OK"
