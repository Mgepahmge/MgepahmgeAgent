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
    通过继承 StructuredTool 创建新对象，不修改原始工具，
    避免 LangChain 内部参数传递问题。
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
    通过重写 invoke/ainvoke 包装缓存逻辑。
    在 LangChain 的公开 API 层面拦截，避免触碰内部 _run 签名问题。
    """
    from langchain_core.tools import StructuredTool
    from langchain_core.callbacks import CallbackManagerForToolRun
    from typing import Any

    original_invoke = tool_obj.invoke
    original_ainvoke = tool_obj.ainvoke

    def _extract_business_args(tool_input) -> dict:
        """从工具输入中提取业务参数（用于缓存 key）"""
        if isinstance(tool_input, dict):
            return tool_input
        elif isinstance(tool_input, str):
            return {"input": tool_input}
        return {}

    def cached_invoke(input: Any, config=None, **kwargs) -> str:
        business_args = _extract_business_args(input)

        if cache.should_cache(tool_obj.name, business_args):
            hit = cache.get(tool_obj.name, business_args)
            if hit is not None:
                return hit

        result = original_invoke(input, config=config, **kwargs)

        if cache.should_cache(tool_obj.name, business_args):
            cache.set(tool_obj.name, business_args, result)
            if tool_obj.name == "run_shell":
                cmd = business_args.get("command", "")
                if not _is_shell_readonly(cmd):
                    cache.invalidate("run_shell")

        return result

    async def cached_ainvoke(input: Any, config=None, **kwargs) -> str:
        business_args = _extract_business_args(input)

        if cache.should_cache(tool_obj.name, business_args):
            hit = cache.get(tool_obj.name, business_args)
            if hit is not None:
                return hit

        result = await original_ainvoke(input, config=config, **kwargs)

        if cache.should_cache(tool_obj.name, business_args):
            cache.set(tool_obj.name, business_args, result)
            if tool_obj.name == "run_shell":
                cmd = business_args.get("command", "")
                if not _is_shell_readonly(cmd):
                    cache.invalidate("run_shell")

        return result

    # 在公开 API 层面替换，不触碰内部 _run
    tool_obj.invoke = cached_invoke
    tool_obj.ainvoke = cached_ainvoke
    return tool_obj
