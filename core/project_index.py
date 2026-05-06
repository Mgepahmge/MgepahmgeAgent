"""
project_index.py - 项目深度感知索引

功能：
  - 用 tree-sitter 解析代码，提取函数、类、结构体等符号
  - 用 Chroma 向量库存储代码片段，支持语义检索
  - 持久化到磁盘，按项目根目录索引
  - 启动时检查文件变化，自动决定是否重建索引

支持语言：Python、C、C++
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 索引持久化根目录
INDEX_ROOT = Path(os.getenv("PROJECT_INDEX_DIR", "/root/agent/data/project_index"))

# 各语言对应的文件扩展名
LANG_EXTENSIONS: dict[str, list[str]] = {
    "python": [".py"],
    "cpp":    [".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"],
    "c":      [".c", ".h"],
}

ALL_EXTENSIONS = {ext for exts in LANG_EXTENSIONS.values() for ext in exts}

# 每种语言支持提取的节点类型（tree-sitter node type）
SYMBOL_NODE_TYPES: dict[str, list[str]] = {
    "python": ["function_definition", "class_definition", "decorated_definition"],
    "cpp":    ["function_definition", "class_specifier", "struct_specifier",
               "namespace_definition", "constructor_or_destructor_definition"],
    "c":      ["function_definition", "struct_specifier", "enum_specifier",
               "type_definition"],
}


# ──────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────

@dataclass
class Symbol:
    """一个代码符号（函数、类等）"""
    name: str
    kind: str          # function / class / struct / namespace / ...
    language: str
    file_path: str
    start_line: int
    end_line: int
    signature: str     # 第一行（函数签名 / 类定义行）
    body: str          # 完整代码片段（用于向量化）

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "language": self.language,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "signature": self.signature,
            "body": self.body,
        }


@dataclass
class IndexMeta:
    """索引元数据，持久化到 meta.json"""
    project_root: str
    created_at: float
    file_hashes: dict[str, str]   # {relative_path: md5}
    symbol_count: int = 0
    chunk_count: int = 0

    def save(self, path: Path):
        path.write_text(json.dumps({
            "project_root": self.project_root,
            "created_at": self.created_at,
            "file_hashes": self.file_hashes,
            "symbol_count": self.symbol_count,
            "chunk_count": self.chunk_count,
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> Optional["IndexMeta"]:
        try:
            d = json.loads(path.read_text())
            return cls(**d)
        except Exception:
            return None


# ──────────────────────────────────────────
# tree-sitter 解析
# ──────────────────────────────────────────

_parsers: dict[str, object] = {}  # 缓存已加载的 parser


def _get_parser(language: str):
    """懒加载 tree-sitter parser"""
    if language in _parsers:
        return _parsers[language]
    try:
        import tree_sitter_python as tspython
        import tree_sitter_c as tsc
        import tree_sitter_cpp as tscpp
        from tree_sitter import Language, Parser

        lang_map = {
            "python": tspython.language(),
            "c":      tsc.language(),
            "cpp":    tscpp.language(),
        }
        if language not in lang_map:
            return None
        parser = Parser(Language(lang_map[language]))
        _parsers[language] = parser
        return parser
    except Exception as e:
        logger.warning(f"tree-sitter [{language}] 加载失败: {e}")
        return None


def _ext_to_lang(ext: str) -> Optional[str]:
    for lang, exts in LANG_EXTENSIONS.items():
        if ext in exts:
            return lang
    return None


def _extract_node_name(node, source_bytes: bytes) -> str:
    """提取符号名称"""
    # 找 identifier 或 name 子节点
    for child in node.children:
        if child.type in ("identifier", "name", "field_identifier",
                          "type_identifier", "destructor_name"):
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    return "<anonymous>"


def _node_kind(node_type: str) -> str:
    mapping = {
        "function_definition": "function",
        "constructor_or_destructor_definition": "function",
        "class_definition": "class",
        "class_specifier": "class",
        "struct_specifier": "struct",
        "namespace_definition": "namespace",
        "enum_specifier": "enum",
        "type_definition": "typedef",
        "decorated_definition": "function",
    }
    return mapping.get(node_type, node_type)


def parse_file_symbols(file_path: Path, language: str) -> list[Symbol]:
    """解析单个文件，返回所有顶层符号"""
    parser = _get_parser(language)
    if parser is None:
        return []

    try:
        source_bytes = file_path.read_bytes()
        tree = parser.parse(source_bytes)
        source_lines = source_bytes.decode("utf-8", errors="replace").splitlines()
    except Exception as e:
        logger.debug(f"解析失败 {file_path}: {e}")
        return []

    symbols = []
    target_types = set(SYMBOL_NODE_TYPES.get(language, []))

    def _walk(node, depth=0):
        if node.type in target_types:
            name = _extract_node_name(node, source_bytes)
            start = node.start_point[0]   # 0-indexed
            end = node.end_point[0]
            signature = source_lines[start] if start < len(source_lines) else ""
            # 限制 body 长度（最多 150 行），避免单个 chunk 过大
            body_lines = source_lines[start:min(end + 1, start + 150)]
            body = "\n".join(body_lines)
            symbols.append(Symbol(
                name=name,
                kind=_node_kind(node.type),
                language=language,
                file_path=str(file_path),
                start_line=start + 1,
                end_line=end + 1,
                signature=signature.strip(),
                body=body,
            ))
            # 不递归进符号内部（避免嵌套类/函数重复）
            return
        for child in node.children:
            _walk(child, depth + 1)

    _walk(tree.root_node)
    return symbols


# ──────────────────────────────────────────
# 文件哈希（用于变化检测）
# ──────────────────────────────────────────

def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def _collect_source_files(root: Path) -> list[Path]:
    """递归收集所有源代码文件，跳过常见的无关目录"""
    skip_dirs = {
        ".git", ".svn", "node_modules", "__pycache__", ".cache",
        "build", "dist", "target", ".venv", "venv", "env",
        ".tox", "coverage", ".idea", ".vscode",
    }
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in ALL_EXTENSIONS:
            # 跳过路径中包含 skip_dirs 的文件
            if not any(part in skip_dirs for part in p.parts):
                files.append(p)
    return sorted(files)


def _project_index_dir(project_root: Path) -> Path:
    """根据项目路径生成唯一的索引目录"""
    key = hashlib.md5(str(project_root.resolve()).encode()).hexdigest()[:16]
    return INDEX_ROOT / key


# ──────────────────────────────────────────
# ProjectIndex 主类
# ──────────────────────────────────────────

class ProjectIndex:
    """
    项目深度索引。
    - 自动检测文件变化，决定是否重建
    - 符号存入 Chroma，支持语义检索和精确查找
    - 每个 ProjectIndex 实例对应一个项目根目录
    """

    def __init__(self, project_root: str):
        self.root = Path(project_root).resolve()
        self._index_dir = _project_index_dir(self.root)
        self._meta_path = self._index_dir / "meta.json"
        self._chroma_dir = self._index_dir / "chroma"
        self._meta: Optional[IndexMeta] = None
        self._collection = None   # Chroma collection，懒加载
        self._embeddings = None   # Embedding 函数

    # ──────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────

    @property
    def is_indexed(self) -> bool:
        return self._meta_path.exists()

    def needs_reindex(self) -> tuple[bool, str]:
        """
        检查是否需要重建索引。
        返回 (需要重建, 原因说明)
        """
        if not self.is_indexed:
            return True, "尚未建立索引"
        meta = IndexMeta.load(self._meta_path)
        if meta is None:
            return True, "索引元数据损坏"

        # 收集当前文件哈希
        current_files = _collect_source_files(self.root)
        current_hashes = {
            str(f.relative_to(self.root)): _file_md5(f)
            for f in current_files
        }
        old_hashes = meta.file_hashes

        # 检查变化
        added = set(current_hashes) - set(old_hashes)
        removed = set(old_hashes) - set(current_hashes)
        modified = {k for k in current_hashes if k in old_hashes
                    and current_hashes[k] != old_hashes[k]}

        if added or removed or modified:
            changes = []
            if added:    changes.append(f"新增 {len(added)} 个文件")
            if removed:  changes.append(f"删除 {len(removed)} 个文件")
            if modified: changes.append(f"修改 {len(modified)} 个文件")
            return True, "、".join(changes)

        self._meta = meta
        return False, "索引已是最新"

    def build(self, progress_cb=None) -> IndexMeta:
        """
        建立完整索引。
        progress_cb: 可选回调 (current, total, message)，用于显示进度
        """
        logger.info(f"开始建立项目索引: {self.root}")
        self._index_dir.mkdir(parents=True, exist_ok=True)

        source_files = _collect_source_files(self.root)
        total = len(source_files)
        logger.info(f"发现 {total} 个源代码文件")

        if progress_cb:
            progress_cb(0, total, f"发现 {total} 个源代码文件，开始解析...")

        # 清空旧的 Chroma 数据
        collection = self._get_collection(reset=True)

        all_symbols: list[Symbol] = []
        file_hashes: dict[str, str] = {}

        for i, file_path in enumerate(source_files):
            lang = _ext_to_lang(file_path.suffix)
            if lang is None:
                continue
            rel = str(file_path.relative_to(self.root))
            file_hashes[rel] = _file_md5(file_path)

            symbols = parse_file_symbols(file_path, lang)
            all_symbols.extend(symbols)

            if progress_cb and (i % 10 == 0 or i == total - 1):
                progress_cb(i + 1, total, f"解析 {rel}（{len(symbols)} 个符号）")

        logger.info(f"共提取 {len(all_symbols)} 个符号，开始向量化...")
        if progress_cb:
            progress_cb(total, total, f"提取 {len(all_symbols)} 个符号，向量化中...")

        # 批量写入 Chroma
        chunk_count = self._upsert_symbols(collection, all_symbols)

        meta = IndexMeta(
            project_root=str(self.root),
            created_at=time.time(),
            file_hashes=file_hashes,
            symbol_count=len(all_symbols),
            chunk_count=chunk_count,
        )
        meta.save(self._meta_path)
        self._meta = meta

        logger.info(f"索引建立完成：{len(all_symbols)} 个符号，{chunk_count} 个 chunk")
        return meta

    def search(self, query: str, k: int = 8,
               language: str | None = None,
               kind: str | None = None) -> list[dict]:
        """
        语义检索代码片段。
        language: 限定语言（python/c/cpp）
        kind: 限定符号类型（function/class/struct 等）
        """
        collection = self._get_collection()
        if collection is None:
            return []

        where: dict = {}
        if language:
            where["language"] = language
        if kind:
            where["kind"] = kind

        try:
            results = collection.query(
                query_texts=[query],
                n_results=min(k, collection.count() or 1),
                where=where if where else None,
                include=["documents", "metadatas", "distances"],
            )
            items = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                items.append({
                    "body": doc,
                    "name": meta.get("name", ""),
                    "kind": meta.get("kind", ""),
                    "language": meta.get("language", ""),
                    "file": meta.get("file_path", ""),
                    "line": meta.get("start_line", 0),
                    "signature": meta.get("signature", ""),
                    "score": round(1 - float(dist), 4),
                })
            return items
        except Exception as e:
            logger.warning(f"代码搜索失败: {e}")
            return []

    def find_symbol(self, name: str, kind: str | None = None) -> list[dict]:
        """精确按名称查找符号（大小写不敏感）"""
        collection = self._get_collection()
        if collection is None:
            return []
        try:
            where: dict = {"name": {"$eq": name}}
            if kind:
                where = {"$and": [where, {"kind": kind}]}
            results = collection.get(
                where=where,
                include=["documents", "metadatas"],
            )
            items = []
            for doc, meta in zip(results["documents"], results["metadatas"]):
                items.append({
                    "body": doc,
                    "name": meta.get("name", ""),
                    "kind": meta.get("kind", ""),
                    "language": meta.get("language", ""),
                    "file": meta.get("file_path", ""),
                    "line": meta.get("start_line", 0),
                    "signature": meta.get("signature", ""),
                })
            return items
        except Exception as e:
            logger.warning(f"符号查找失败: {e}")
            return []

    def get_file_outline(self, file_path: str) -> list[dict]:
        """列出单个文件的所有符号（函数、类等）"""
        collection = self._get_collection()
        if collection is None:
            return []
        try:
            results = collection.get(
                where={"file_path": file_path},
                include=["metadatas"],
            )
            items = []
            for meta in results["metadatas"]:
                items.append({
                    "name": meta.get("name", ""),
                    "kind": meta.get("kind", ""),
                    "line": meta.get("start_line", 0),
                    "signature": meta.get("signature", ""),
                })
            return sorted(items, key=lambda x: x["line"])
        except Exception as e:
            logger.warning(f"获取文件大纲失败: {e}")
            return []

    def status(self) -> dict:
        """返回当前索引状态"""
        if not self.is_indexed:
            return {"状态": "未索引", "项目": str(self.root)}
        meta = self._meta or IndexMeta.load(self._meta_path)
        if meta is None:
            return {"状态": "元数据损坏", "项目": str(self.root)}
        import time as _time
        age = _time.time() - meta.created_at
        age_str = (f"{int(age/3600)}小时前" if age > 3600
                   else f"{int(age/60)}分钟前" if age > 60
                   else "刚刚")
        return {
            "状态": "已索引",
            "项目": str(self.root),
            "符号数": meta.symbol_count,
            "文件数": len(meta.file_hashes),
            "建立时间": age_str,
            "索引目录": str(self._index_dir),
        }

    # ──────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────

    def _get_embeddings(self):
        """获取 Embedding 函数（复用 RAG 的本地模型）"""
        if self._embeddings is not None:
            return self._embeddings
        try:
            import chromadb.utils.embedding_functions as ef
            from core.config import config
            model_name = config.rag.embedding_model or "BAAI/bge-m3"
            cache_dir = os.getenv("EMBEDDING_CACHE_DIR",
                                   str(Path(__file__).parent.parent / "data" / "models"))
            device = config.rag.embedding_device or "cuda"
            self._embeddings = ef.SentenceTransformerEmbeddingFunction(
                model_name=model_name,
                device=device,
                cache_folder=cache_dir,
            )
        except Exception as e:
            logger.warning(f"Embedding 加载失败，使用默认: {e}")
            import chromadb.utils.embedding_functions as ef
            self._embeddings = ef.DefaultEmbeddingFunction()
        return self._embeddings

    def _get_collection(self, reset: bool = False):
        """获取 Chroma collection（懒加载）"""
        if self._collection is not None and not reset:
            return self._collection
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(self._chroma_dir))
            if reset:
                try:
                    client.delete_collection("symbols")
                except Exception:
                    pass
            self._collection = client.get_or_create_collection(
                name="symbols",
                embedding_function=self._get_embeddings(),
                metadata={"hnsw:space": "cosine"},
            )
            return self._collection
        except Exception as e:
            logger.error(f"Chroma 初始化失败: {e}")
            return None

    def _upsert_symbols(self, collection, symbols: list[Symbol]) -> int:
        """批量写入符号到 Chroma，返回写入数量"""
        if not symbols:
            return 0
        BATCH = 100
        count = 0
        for i in range(0, len(symbols), BATCH):
            batch = symbols[i:i + BATCH]
            ids = [
                hashlib.md5(
                    f"{s.file_path}:{s.start_line}:{s.name}".encode()
                ).hexdigest()
                for s in batch
            ]
            docs = [s.body for s in batch]
            metas = [
                {k: v for k, v in s.to_dict().items() if k != "body"}
                for s in batch
            ]
            try:
                collection.upsert(ids=ids, documents=docs, metadatas=metas)
                count += len(batch)
            except Exception as e:
                logger.warning(f"批量写入失败 (batch {i}): {e}")
        return count


# ──────────────────────────────────────────
# 全局索引管理（单个 Agent 同时只索引一个项目）
# ──────────────────────────────────────────

_current_index: Optional[ProjectIndex] = None


def get_current_index() -> Optional[ProjectIndex]:
    return _current_index


def set_current_index(index: ProjectIndex):
    global _current_index
    _current_index = index
