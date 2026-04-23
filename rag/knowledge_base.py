"""
rag.py - RAG 知识库
- 本地 Embedding（CUDA 加速，bge-m3）
- pgvector 向量存储
- 支持 PDF / TXT / DOCX / 代码文件导入
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _make_embeddings(cfg):
    """构造本地 Embedding 模型（CUDA 加速）"""
    import os
    import torch

    device = cfg.embedding_device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，回退到 CPU")
        device = "cpu"

    # 固定缓存路径：优先用 EMBEDDING_CACHE_DIR，否则放到项目 data/models/
    cache_dir = os.getenv("EMBEDDING_CACHE_DIR", "")
    if not cache_dir:
        cache_dir = str(Path(__file__).parent.parent / "data" / "models")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # 模型已下载时强制离线，跳过网络检查，加速启动
    model_path = Path(cache_dir) / f"models--{cfg.embedding_model.replace('/', '--')}"
    if model_path.exists():
        os.environ["HF_HUB_OFFLINE"] = "1"
        logger.info(f"Embedding 模型已缓存，使用离线模式")
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        logger.info(f"首次下载 Embedding 模型 {cfg.embedding_model}")

    logger.info(f"加载 Embedding 模型 {cfg.embedding_model} @ {device}，缓存路径: {cache_dir}")

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=cfg.embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
        cache_folder=cache_dir,
    )


class KnowledgeBase:
    """
    对外接口：
      kb = KnowledgeBase.connect(cfg)
      kb.ingest("/path/to/docs")
      docs = kb.search("查询语句", k=5)
    """

    def __init__(self, store, embeddings):
        self._store = store
        self._embeddings = embeddings

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------
    @classmethod
    def connect(cls, cfg) -> Optional["KnowledgeBase"]:
        """连接 pgvector；失败时返回 None（允许无 RAG 运行）"""
        try:
            from langchain_postgres import PGVector
            embeddings = _make_embeddings(cfg)
            store = PGVector(
                embeddings=embeddings,
                collection_name=cfg.table,
                connection=cfg.connection_string,
                use_jsonb=True,
            )
            logger.info("RAG 知识库连接成功")
            return cls(store, embeddings)
        except Exception as e:
            logger.warning(f"RAG 连接失败（将以无知识库模式运行）: {e}")
            return None

    # ------------------------------------------------------------------
    # 文档导入
    # ------------------------------------------------------------------
    def ingest(self, path: str, chunk_size: int = 800, chunk_overlap: int = 150) -> int:
        """
        导入文件或目录，返回写入的 chunk 数量。
        支持：.pdf .txt .md .py .cpp .java .docx .json .yaml
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import (
            PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader,
        )

        p = Path(path).expanduser()
        files = list(p.rglob("*")) if p.is_dir() else [p]

        LOADER_MAP = {
            ".pdf":  PyPDFLoader,
            ".docx": UnstructuredWordDocumentLoader,
        }
        TEXT_EXTS = {".txt", ".md", ".py", ".cpp", ".c", ".h", ".java",
                     ".json", ".yaml", ".yml", ".sql", ".sh", ".toml"}

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        all_chunks = []
        for f in files:
            ext = f.suffix.lower()
            try:
                if ext in LOADER_MAP:
                    docs = LOADER_MAP[ext](str(f)).load()
                elif ext in TEXT_EXTS:
                    docs = TextLoader(str(f), autodetect_encoding=True).load()
                else:
                    continue
                chunks = splitter.split_documents(docs)
                # 附加来源元数据
                for c in chunks:
                    c.metadata["source"] = str(f)
                all_chunks.extend(chunks)
                logger.debug(f"  {f.name}: {len(chunks)} chunks")
            except Exception as e:
                logger.warning(f"  跳过 {f.name}: {e}")

        if all_chunks:
            self._store.add_documents(all_chunks)
            logger.info(f"导入完成：{len(all_chunks)} chunks from {len(files)} files")
        return len(all_chunks)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 5) -> list[dict]:
        """返回最相关的 k 个文档片段"""
        results = self._store.similarity_search_with_relevance_scores(query, k=k)
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "score": round(float(score), 4),
            }
            for doc, score in results
        ]

    def as_retriever(self, k: int = 5):
        return self._store.as_retriever(search_kwargs={"k": k})
