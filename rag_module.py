# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import json
import hashlib
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

# Dependencies: langchain / chroma / sentence-transformers
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    # Optional: semantic chunking (used in your demo)
    from langchain_experimental.text_splitter import SemanticChunker  # type: ignore
    _HAS_SEMANTIC_CHUNKER = True
except Exception:
    _HAS_SEMANTIC_CHUNKER = False

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from sentence_transformers import CrossEncoder


# ======================
# Utilities
# ======================
def _safe_read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _detect_content_type(text: str) -> str:
    """
    Heuristic content type detection for dynamic chunk sizing.
    """
    if re.search(r"\b(def |class |import |#include|printf\(|print\()", text):
        return "code"
    if re.search(r"\|.+\|", text) and re.search(r"%|\bGy\b|\bcGy\b|\bD90\b|\bV100\b", text):
        return "table"
    return "normal"


def _fingerprint_dir(root: str, exts: Tuple[str, ...] = (".pdf", ".txt", ".md")) -> str:
    """
    Compute a fingerprint for the knowledge_base directory based on:
    - relative path
    - mtime
    - file size
    """
    items: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith(exts):
                continue
            p = os.path.join(dirpath, fn)
            try:
                st = os.stat(p)
                rel = os.path.relpath(p, root)
                items.append(f"{rel}|{int(st.st_mtime)}|{st.st_size}")
            except Exception:
                continue
    items.sort()
    h = hashlib.sha256("\n".join(items).encode("utf-8", errors="ignore")).hexdigest()
    return h


# ======================
# Config
# ======================
@dataclass
class RAGConfig:
    knowledge_base_dir: str = "/hdd1/xz/deepseek/knowledge_base"
    chroma_dir: str = "/hdd1/xz/deepseek/vector_db"
    embed_model_path: str = "/hdd1/xz/models/BAAI/bge-small-zh-v1.5"
    reranker_model_path: str = "/hdd1/xz/models/BAAI/bge-reranker-large"
    embed_device: str = "cuda"
    reranker_device: str = "cuda:1"
    bm25_k: int = 8
    vector_k: int = 8
    top_k: int = 3
    min_score_threshold: float = 0.3

    # Splitter config
    use_semantic_chunker: bool = True
    semantic_breakpoint_threshold_amount: int = 82

    # Fallback splitter config
    chunk_size_normal: int = 512
    chunk_overlap_normal: int = 128
    chunk_size_code: int = 256
    chunk_overlap_code: int = 64
    chunk_size_table: int = 384
    chunk_overlap_table: int = 96

    # Prompt context cap
    max_context_chars: int = 6000


# ======================
# Document processing
# ======================
class SmartDocumentProcessor:
    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self.embed_model = HuggingFaceEmbeddings(
            model_name=cfg.embed_model_path,
            model_kwargs={"device": cfg.embed_device},
            encode_kwargs={"batch_size": 16},
        )

    def load_documents(self) -> List[Any]:
        if not os.path.isdir(self.cfg.knowledge_base_dir):
            raise FileNotFoundError(f"knowledge_base_dir not found: {self.cfg.knowledge_base_dir}")

        loaders = [
            DirectoryLoader(self.cfg.knowledge_base_dir, glob="**/*.pdf", loader_cls=PyPDFLoader),
            DirectoryLoader(self.cfg.knowledge_base_dir, glob="**/*.txt", loader_cls=TextLoader),
            DirectoryLoader(self.cfg.knowledge_base_dir, glob="**/*.md", loader_cls=TextLoader),
        ]
        documents: List[Any] = []
        for loader in loaders:
            documents.extend(loader.load())
        return documents

    def split_documents(self, documents: List[Any]) -> List[Any]:
        """
        Two-stage chunking:
        1) Semantic chunker if available, otherwise a recursive splitter
        2) Re-split each chunk with dynamic sizes based on content type
        """
        base_chunks: List[Any] = []
        if self.cfg.use_semantic_chunker and _HAS_SEMANTIC_CHUNKER:
            chunker = SemanticChunker(
                embeddings=self.embed_model,
                breakpoint_threshold_amount=self.cfg.semantic_breakpoint_threshold_amount,
                add_start_index=True,
            )
            base_chunks = chunker.split_documents(documents)
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.cfg.chunk_size_normal,
                chunk_overlap=self.cfg.chunk_overlap_normal,
            )
            base_chunks = splitter.split_documents(documents)

        final_chunks: List[Any] = []
        for chunk in base_chunks:
            ctype = _detect_content_type(chunk.page_content)
            if ctype == "code":
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.cfg.chunk_size_code,
                    chunk_overlap=self.cfg.chunk_overlap_code,
                )
            elif ctype == "table":
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.cfg.chunk_size_table,
                    chunk_overlap=self.cfg.chunk_overlap_table,
                )
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.cfg.chunk_size_normal,
                    chunk_overlap=self.cfg.chunk_overlap_normal,
                )
            final_chunks.extend(splitter.split_documents([chunk]))

        # Fill metadata
        for i, ch in enumerate(final_chunks):
            ch.metadata = ch.metadata or {}
            ch.metadata.update(
                {
                    "chunk_id": f"chunk_{i}",
                    "content_type": _detect_content_type(ch.page_content),
                }
            )
        return final_chunks


# ======================
# Hybrid retriever (Vector + BM25 + rerank)
# ======================
class HybridRetriever:
    def __init__(self, cfg: RAGConfig, chunks: List[Any]):
        self.cfg = cfg

        # Use the same embedding model for vector DB
        self.embedding = HuggingFaceEmbeddings(
            model_name=cfg.embed_model_path,
            model_kwargs={"device": cfg.embed_device},
        )

        # Vector DB (persistent)
        self.vector_db = Chroma.from_documents(
            chunks,
            embedding=self.embedding,
            persist_directory=cfg.chroma_dir,
        )

        # BM25
        self.bm25_retriever = BM25Retriever.from_documents(chunks, k=int(cfg.bm25_k))

        # Hybrid ensemble
        self.ensemble = EnsembleRetriever(
            retrievers=[
                self.vector_db.as_retriever(search_kwargs={"k": int(cfg.vector_k)}),
                self.bm25_retriever,
            ],
            weights=[0.6, 0.4],
        )

        # Cross-encoder reranker
        self.reranker = CrossEncoder(cfg.reranker_model_path, device=cfg.reranker_device)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score_threshold: Optional[float] = None,
    ) -> List[Any]:
        top_k = int(top_k if top_k is not None else self.cfg.top_k)
        thr = float(min_score_threshold if min_score_threshold is not None else self.cfg.min_score_threshold)

        docs = self.ensemble.get_relevant_documents(query)
        if not docs:
            return []

        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)

        ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
        filtered = [(d, s) for d, s in ranked if float(s) >= thr]
        return [d for d, _ in filtered[:top_k]]


# ======================
# Context provider
# ======================
class RAGContextProvider:
    """
    Provide a compact context string for LLM prompts.
    - Caches the knowledge base fingerprint in _rag_meta.json
    - If fingerprint matches, tries to reuse persisted Chroma DB
      while rebuilding BM25 from current documents (simple and robust)
    """

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self._retriever: Optional[HybridRetriever] = None
        self._kb_fingerprint: Optional[str] = None

        os.makedirs(cfg.chroma_dir, exist_ok=True)
        self._init_or_load()

    def _meta_path(self) -> str:
        return os.path.join(self.cfg.chroma_dir, "_rag_meta.json")

    def _load_meta(self) -> Dict[str, Any]:
        p = self._meta_path()
        if os.path.exists(p):
            try:
                return json.loads(_safe_read_text(p))
            except Exception:
                return {}
        return {}

    def _save_meta(self, meta: Dict[str, Any]) -> None:
        p = self._meta_path()
        with open(p, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _init_or_load(self) -> None:
        kb_dir = self.cfg.knowledge_base_dir
        fp = _fingerprint_dir(kb_dir)
        meta = self._load_meta()
        old_fp = meta.get("kb_fingerprint", "")

        can_load = (fp == old_fp)
        if can_load:
            try:
                # We still rebuild chunks for BM25 (needs raw docs), but Chroma is persisted.
                processor = SmartDocumentProcessor(self.cfg)
                docs = processor.load_documents()
                chunks = processor.split_documents(docs)

                self._retriever = HybridRetriever(self.cfg, chunks)
                self._kb_fingerprint = fp
                self._save_meta({"kb_fingerprint": fp, "note": "loaded_or_rebuilt_bm25"})
                return
            except Exception:
                pass

        # Rebuild everything
        processor = SmartDocumentProcessor(self.cfg)
        docs = processor.load_documents()
        chunks = processor.split_documents(docs)
        self._retriever = HybridRetriever(self.cfg, chunks)
        self._kb_fingerprint = fp
        self._save_meta({"kb_fingerprint": fp, "note": "rebuilt_all"})

    def build_context_text(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score_threshold: Optional[float] = None,
    ) -> str:
        if self._retriever is None:
            return "<RAG retriever not initialized>"

        docs = self._retriever.retrieve(query, top_k=top_k, min_score_threshold=min_score_threshold)
        if not docs:
            return "<RAG: no relevant context found>"

        parts: List[str] = []
        for doc in docs:
            src = doc.metadata.get("source", "<unknown>")
            ctype = doc.metadata.get("content_type", "<unknown>")
            text = (doc.page_content or "").strip()
            parts.append(f"[Source: {src} | Type: {ctype}]\n{text}")

        context = "\n\n".join(parts)

        # Truncate to avoid blowing up the prompt
        if len(context) > int(self.cfg.max_context_chars):
            context = context[: int(self.cfg.max_context_chars)] + "\n\n... <RAG context truncated>"
        return context
