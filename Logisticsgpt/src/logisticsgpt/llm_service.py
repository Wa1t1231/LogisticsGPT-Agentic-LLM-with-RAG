"""Load a quantised SOTA chat model and expose a simple RAG `ask_rag()` API."""

from __future__ import annotations
from pathlib import Path
import logging

from llama_cpp import Llama
from llama_index import VectorStoreIndex, StorageContext
from llama_index.llms import LlamaCPP
from llama_index.retrievers import VectorIndexRetriever

from .config import VECTOR_DB_PATH

LOGGER = logging.getLogger(__name__)
MODEL_PATH_DEFAULT = Path(
    # You can swap in DeepSeek, Grok, Claude quantised GGUF here
    "models/deepseek-llm-7b-chat-q4_K_M.gguf"
)


class RAGService:
    """Singleton-like wrapper holding the quantised LLM and the vector index."""

    def __init__(self, model_path: str | Path | None = None):
        model_path = Path(model_path or MODEL_PATH_DEFAULT)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Quantised model not found at {model_path}. Please download GGUF file."
            )

        # Load quantised model via llama.cpp bindings (CPU/GPU mixed)
        self._llama_raw = Llama(
            model_path=str(model_path),
            n_gpu_layers=40,  # tweak according to your GPU
            n_ctx=8192,
            logits_all=False,
        )
        self.llm = LlamaCPP(
            model_path=str(model_path),
            context_window=8192,
            max_new_tokens=1024,
            temperature=0.1,
        )

        # Load existing Qdrant vector store
        storage_ctx = StorageContext.from_defaults(persist_dir=str(VECTOR_DB_PATH))
        self.index = VectorStoreIndex.from_persist_dir(
            persist_dir=str(VECTOR_DB_PATH), storage_context=storage_ctx
        )
        self.retriever: VectorIndexRetriever = self.index.as_retriever(similarity_top_k=8)

    # ---------------- public API ----------------
    def ask(self, query: str) -> str:
        """Retrieve‑and‑generate answer for user `query`."""
        context_nodes = self.retriever.retrieve(query)
        context_text = "\n".join(node.node.text for node in context_nodes)

        system_prompt = (
            "You are LogisticsGPT, a meticulous logistics analyst. You answer using data from the "
            "context provided. If answer is not in context, say 'I don't know'."
"
        )

        prompt = f"""{system_prompt}
### Context
{context_text}

### Question
{query}

### Answer
"""

        response = self.llm.complete(prompt)
        return response.text.strip()


# Global instance
_rag_service: RAGService | None = None


def _get_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def ask_rag(question: str) -> str:
    """Module level helper to answer questions via RAG."""
    return _get_service().ask(question)
