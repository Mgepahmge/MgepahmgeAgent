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

    # 禁用 sentence-transformers 的 tqdm 进度条
    # 通过环境变量 TQDM_DISABLE 全局禁用，比替换 tqdm.tqdm 更可靠
    import os
    _prev = os.environ.get("TQDM_DISABLE", "")
    os.environ["TQDM_DISABLE"] = "1"

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=cfg.embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
            cache_folder=cache_dir,
        )
    finally:
        if _prev:
            os.environ["TQDM_DISABLE"] = _prev
        else:
            os.environ.pop("TQDM_DISABLE", None)

    return embeddings


class KnowledgeBase:
    """
    对外接口：
      kb = KnowledgeBase.connect(cfg)
      kb.ingest("/path/to/docs")
      docs = kb.search("查询语句", k=5)
    """

    # 状态常量
    STATE_DISCONNECTED = "未连接"
    STATE_INITIALIZING = "初始化中"
    STATE_READY        = "初始化完成"

    def __init__(self, cfg):
        self._cfg = cfg
        self._store = None
        self._embeddings = None
        self._state = self.STATE_INITIALIZING
        self._init_error: str = ""
        self._init_thread = None

    @property
    def state(self) -> str:
        return self._state

    def _background_init(self):
        """后台线程：执行 PGVector 初始化（含建表/扩展操作）"""
        try:
            from langchain_postgres import PGVector
            self._embeddings = _make_embeddings(self._cfg)
            self._store = PGVector(
                embeddings=self._embeddings,
                collection_name=self._cfg.table,
                connection=self._cfg.connection_string,
                use_jsonb=True,
            )
            self._state = self.STATE_READY
            logger.info("RAG 向量存储初始化完成")
        except Exception as e:
            self._state = self.STATE_DISCONNECTED
            self._init_error = str(e)
            logger.warning(f"RAG 向量存储初始化失败: {e}")

    def _ensure_store(self):
        """等待后台初始化完成，未就绪时打印提示并阻塞"""
        if self._state == self.STATE_READY:
            return True
        if self._state == self.STATE_DISCONNECTED:
            return False
        # 初始化中：等待完成
        print("⏳ RAG 知识库初始化中，请稍候...", flush=True)
        if self._init_thread and self._init_thread.is_alive():
            self._init_thread.join()
        return self._state == self.STATE_READY

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------
    @classmethod
    def connect(cls, cfg) -> Optional["KnowledgeBase"]:
        """
        启动时 ping 数据库验证连接，成功后在后台线程异步初始化 PGVector。
        """
        try:
            import psycopg2
            import threading
            conn = psycopg2.connect(
                host=cfg.host, port=cfg.port, dbname=cfg.db,
                user=cfg.user, password=cfg.password,
                connect_timeout=5,
            )
            conn.close()
            logger.info("RAG 数据库连接成功，后台初始化向量存储...")
            kb = cls(cfg)
            t = threading.Thread(target=kb._background_init, daemon=True)
            t.start()
            kb._init_thread = t
            return kb
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
            if not self._ensure_store():
                logger.error("RAG 存储不可用，导入失败")
                return 0
            self._store.add_documents(all_chunks)
            logger.info(f"导入完成：{len(all_chunks)} chunks from {len(files)} files")
        return len(all_chunks)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 5) -> list[dict]:
        """返回最相关的 k 个文档片段"""
        if not self._ensure_store():
            return []
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
        if not self._ensure_store():
            return None
        return self._store.as_retriever(search_kwargs={"k": k})
