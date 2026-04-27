"""
tool_cache.py - 工具调用结果缓存

设计原则：
  - 缓存范围：单个 AgentRuntime 实例内（不跨 session，不跨 Agent）
  - 缓存 key：(工具名, 参数的规范化哈希)
  - 每种工具独立的 TTL 和缓存策略
  - 写操作不缓存，且会使相关缓存失效

缓存策略：
  工具             策略       TTL
  ────────────────────────────────────────
  web_search       缓存       600s（10分钟）
  fetch_url        缓存       300s（5分钟）
  run_shell        条件缓存   30s（只读命令）
  其他工具          透传       不缓存
"""
from __future__ import annotations
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# 默认 TTL 配置（秒）
_DEFAULT_TTL: dict[str, int] = {
    "web_search": 600,
    "fetch_url":  300,
    "run_shell":  30,
}

# run_shell 中不缓存的命令前缀（写/破坏性操作）
_SHELL_WRITE_PREFIXES = {
    "rm", "rmdir", "mv", "cp", "mkdir", "touch", "chmod", "chown",
    "apt", "apt-get", "pip", "pip3", "conda", "npm", "yarn",
    "systemctl", "service", "kill", "pkill",
    "tee", "truncate", "dd",
    "git commit", "git push", "git reset", "git rm",
    "psql", "mysql",       # SQL 写操作（粗粒度判断）
}


def _is_shell_readonly(command: str) -> bool:
    """判断 shell 命令是否为只读操作（保守策略：不确定时返回 False）"""
    cmd = command.strip().lower()
    first_token = cmd.split()[0].split("/")[-1] if cmd.split() else ""
    if first_token in _SHELL_WRITE_PREFIXES:
        return False
    # 包含重定向写入符号
    if any(op in command for op in [" > ", " >> ", "| tee", "| dd"]):
        return False
    # 包含写操作关键词
    if any(kw in cmd for kw in ["write", "insert", "update", "delete", "drop", "create table"]):
        return False
    return True


@dataclass
class _CacheEntry:
    value: str
    expires_at: float

    def is_valid(self) -> bool:
        return time.monotonic() < self.expires_at


class ToolCache:
    """
    工具调用缓存，绑定到单个 AgentRuntime 实例。
    非线程安全（每个 AgentRuntime 有独立实例，各自在自己的事件循环里调用）。
    """

    def __init__(self):
        self._store: dict[str, _CacheEntry] = {}
        self._hits: int = 0
        self._misses: int = 0

    # ──────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────

    def get(self, tool_name: str, kwargs: dict) -> str | None:
        """查询缓存，命中返回结果字符串，未命中返回 None"""
        key = self._make_key(tool_name, kwargs)
        entry = self._store.get(key)
        if entry and entry.is_valid():
            self._hits += 1
            logger.debug(f"缓存命中: {tool_name}({self._repr_kwargs(kwargs)})")
            return entry.value
        if entry:
            del self._store[key]   # 清理过期条目
        self._misses += 1
        return None

    def set(self, tool_name: str, kwargs: dict, value: str):
        """写入缓存"""
        ttl = _DEFAULT_TTL.get(tool_name, 0)
        if ttl <= 0:
            return
        key = self._make_key(tool_name, kwargs)
        self._store[key] = _CacheEntry(
            value=value,
            expires_at=time.monotonic() + ttl,
        )
        logger.debug(f"缓存写入: {tool_name}({self._repr_kwargs(kwargs)}) TTL={ttl}s")

    def should_cache(self, tool_name: str, kwargs: dict) -> bool:
        """判断此次调用是否应该缓存"""
        if tool_name not in _DEFAULT_TTL:
            return False
        if tool_name == "run_shell":
            command = kwargs.get("command", "")
            return _is_shell_readonly(command)
        return True

    def invalidate(self, pattern: str | None = None):
        """
        使缓存失效。
        pattern=None  : 清空全部
        pattern='run_shell' : 清空指定工具的缓存
        """
        if pattern is None:
            count = len(self._store)
            self._store.clear()
            logger.debug(f"缓存清空: {count} 条")
        else:
            keys = [k for k in self._store if k.startswith(f"{pattern}:")]
            for k in keys:
                del self._store[k]
            logger.debug(f"缓存失效: {pattern} {len(keys)} 条")

    def stats(self) -> dict:
        """返回缓存统计信息"""
        valid = sum(1 for e in self._store.values() if e.is_valid())
        total = self._hits + self._misses
        hit_rate = f"{self._hits/total*100:.1f}%" if total > 0 else "N/A"
        return {
            "有效条目": valid,
            "总条目": len(self._store),
            "命中次数": self._hits,
            "未命中次数": self._misses,
            "命中率": hit_rate,
        }

    # ──────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────

    @staticmethod
    def _make_key(tool_name: str, kwargs: dict) -> str:
        """生成稳定的缓存 key"""
        canonical = json.dumps(kwargs, sort_keys=True, ensure_ascii=False, default=str)
        digest = hashlib.md5(canonical.encode()).hexdigest()[:12]
        return f"{tool_name}:{digest}"

    @staticmethod
    def _repr_kwargs(kwargs: dict) -> str:
        """简短展示参数（用于日志）"""
        s = json.dumps(kwargs, ensure_ascii=False, default=str)
        return s[:60] + "..." if len(s) > 60 else s


def wrap_tools_with_cache(tools: list, cache: ToolCache) -> list:
    """
    用缓存包装器包装工具列表。
    对每个支持缓存的工具，替换其 _run 方法加入缓存逻辑。
    不修改原始工具对象，返回包装后的新列表。
    """
    wrapped = []
    for t in tools:
        if t.name in _DEFAULT_TTL:
            wrapped.append(_wrap_one(t, cache))
        else:
            wrapped.append(t)
    return wrapped


def _wrap_one(tool_obj, cache: ToolCache):
    """
    包装单个工具，在调用前查缓存，调用后写缓存。
    保留原始工具的所有属性（name, description, _source 等）。
    """
    original_run = tool_obj._run

    def cached_run(*args, **kwargs) -> str:
        # LangChain 工具调用时参数通过 kwargs 传入
        tool_kwargs = kwargs if kwargs else {}
        if args:
            # positional args 的情况：尝试从 tool schema 映射
            try:
                import inspect
                sig = inspect.signature(original_run)
                params = list(sig.parameters.keys())
                for i, a in enumerate(args):
                    if i < len(params):
                        tool_kwargs[params[i]] = a
            except Exception:
                pass

        if cache.should_cache(tool_obj.name, tool_kwargs):
            cached = cache.get(tool_obj.name, tool_kwargs)
            if cached is not None:
                return cached

        result = original_run(*args, **kwargs)

        if cache.should_cache(tool_obj.name, tool_kwargs):
            cache.set(tool_obj.name, tool_kwargs, result)
            # 写操作后使 run_shell 缓存失效
            if tool_obj.name != "run_shell":
                pass
            elif not _is_shell_readonly(tool_kwargs.get("command", "")):
                cache.invalidate("run_shell")

        return result

    # 替换 _run，保留其他所有属性
    tool_obj._run = cached_run
    return tool_obj
