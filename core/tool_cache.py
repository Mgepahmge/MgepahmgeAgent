"""
tool_cache.py - 工具调用结果缓存

实现方式：
  不修改工具对象本身，而是包装 LangGraph 的 ToolNode，
  在节点层面拦截工具调用消息，查缓存/写缓存后再决定是否真正执行。

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

logger = logging.getLogger(__name__)

# 各工具默认 TTL（秒），不在此表中的工具不缓存
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
    "psql", "mysql",
}


def _is_shell_readonly(command: str) -> bool:
    """判断 shell 命令是否为只读操作（保守策略）"""
    cmd = command.strip().lower()
    first_token = cmd.split()[0].split("/")[-1] if cmd.split() else ""
    if first_token in _SHELL_WRITE_PREFIXES:
        return False
    if any(op in command for op in [" > ", " >> ", "| tee", "| dd"]):
        return False
    if any(kw in cmd for kw in ["insert ", "update ", "delete ", "drop ", "create table"]):
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
    """

    def __init__(self):
        self._store: dict[str, _CacheEntry] = {}
        self._hits: int = 0
        self._misses: int = 0

    def should_cache(self, tool_name: str, tool_args: dict) -> bool:
        if tool_name not in _DEFAULT_TTL:
            return False
        if tool_name == "run_shell":
            return _is_shell_readonly(tool_args.get("command", ""))
        return True

    def get(self, tool_name: str, tool_args: dict) -> str | None:
        key = self._make_key(tool_name, tool_args)
        entry = self._store.get(key)
        if entry and entry.is_valid():
            self._hits += 1
            logger.debug(f"缓存命中: {tool_name}")
            return entry.value
        if entry:
            del self._store[key]
        self._misses += 1
        return None

    def set(self, tool_name: str, tool_args: dict, value: str):
        ttl = _DEFAULT_TTL.get(tool_name, 0)
        if ttl <= 0:
            return
        key = self._make_key(tool_name, tool_args)
        self._store[key] = _CacheEntry(
            value=value,
            expires_at=time.monotonic() + ttl,
        )
        logger.debug(f"缓存写入: {tool_name} TTL={ttl}s")

    def invalidate_shell(self):
        """写操作后清空 run_shell 的缓存"""
        keys = [k for k in self._store if k.startswith("run_shell:")]
        for k in keys:
            del self._store[k]

    def invalidate_all(self):
        count = len(self._store)
        self._store.clear()
        logger.debug(f"缓存全部清空: {count} 条")

    def stats(self) -> dict:
        valid = sum(1 for e in self._store.values() if e.is_valid())
        total = self._hits + self._misses
        hit_rate = f"{self._hits / total * 100:.1f}%" if total > 0 else "N/A"
        return {
            "有效条目": valid,
            "总条目": len(self._store),
            "命中次数": self._hits,
            "未命中次数": self._misses,
            "命中率": hit_rate,
        }

    @staticmethod
    def _make_key(tool_name: str, tool_args: dict) -> str:
        canonical = json.dumps(tool_args, sort_keys=True,
                               ensure_ascii=False, default=str)
        digest = hashlib.md5(canonical.encode()).hexdigest()[:12]
        return f"{tool_name}:{digest}"


# ──────────────────────────────────────────
# 缓存化 ToolNode
# ──────────────────────────────────────────

def make_cached_tool_node(tools: list, cache: ToolCache):
    """
    返回一个带缓存的 ToolNode 替代品。
    在 LangGraph 节点层面拦截工具调用消息，
    命中缓存时直接构造 ToolMessage 返回，跳过真正的工具执行。
    未命中时执行原始工具并写入缓存。
    """
    from langgraph.prebuilt import ToolNode
    from langchain_core.messages import ToolMessage, AIMessage

    tool_map = {t.name: t for t in tools}
    original_node = ToolNode(tools)

    async def cached_node(state: dict) -> dict:
        messages = state.get("messages", [])
        last = messages[-1] if messages else None

        # 只处理有 tool_calls 的 AIMessage
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return await original_node.ainvoke(state)

        cached_results: list[ToolMessage] = []
        uncached_calls: list = []

        for call in last.tool_calls:
            tool_name = call.get("name", "")
            tool_args = call.get("args", {})
            call_id = call.get("id", "")

            if cache.should_cache(tool_name, tool_args):
                hit = cache.get(tool_name, tool_args)
                if hit is not None:
                    cached_results.append(
                        ToolMessage(content=hit, tool_call_id=call_id)
                    )
                    continue
            uncached_calls.append(call)

        # 全部命中缓存：直接返回
        if not uncached_calls:
            return {"messages": cached_results}

        # 部分或全部未命中：执行原始节点
        # 构造只含未命中调用的临时状态
        if len(uncached_calls) < len(last.tool_calls):
            import copy
            temp_last = copy.copy(last)
            object.__setattr__(temp_last, "tool_calls", uncached_calls)
            temp_state = {**state, "messages": messages[:-1] + [temp_last]}
        else:
            temp_state = state

        result = await original_node.ainvoke(temp_state)
        new_messages = result.get("messages", [])

        # 将未命中的结果写入缓存
        call_map = {c.get("id", ""): c for c in uncached_calls}
        for msg in new_messages:
            if isinstance(msg, ToolMessage):
                call = call_map.get(msg.tool_call_id, {})
                tool_name = call.get("name", "")
                tool_args = call.get("args", {})
                if cache.should_cache(tool_name, tool_args):
                    cache.set(tool_name, tool_args, msg.content)
                    # 写操作后清空 shell 缓存
                    if tool_name == "run_shell":
                        if not _is_shell_readonly(tool_args.get("command", "")):
                            cache.invalidate_shell()

        # 合并缓存命中 + 真实执行的结果
        return {"messages": cached_results + new_messages}

    return cached_node
