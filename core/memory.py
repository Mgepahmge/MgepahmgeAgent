"""
memory.py - 长期记忆管理
- 自动从对话提取记忆
- 提供给 Agent 的记忆工具（save/delete/list）
- 构建注入 system prompt 的记忆片段
"""
from __future__ import annotations
import json
import logging
from .database import load_all_memories, save_memory, delete_memory

logger = logging.getLogger(__name__)

MEMORY_EXTRACT_THRESHOLD = 10


def build_memory_prompt() -> str:
    memories = load_all_memories()
    if not memories:
        return ""
    lines = [f"- {m['key']}: {m['value']}" for m in memories]
    return "【长期记忆】\n" + "\n".join(lines)


async def extract_memories(messages: list, llm) -> int:
    """从对话中自动提取长期记忆"""
    if not messages:
        return 0
    conversation = "\n".join(
        f"{m['role']}: {m['content'][:500]}"
        for m in messages[-20:]
        if m['role'] in ('human', 'user', 'assistant', 'ai')
    )
    prompt = f"""分析以下对话，提取值得长期记住的用户信息（偏好、习惯、重要事实等）。
只提取确定的、有价值的信息，不要猜测。

对话内容：
{conversation}

返回格式（JSON 数组，如无有价值信息则返回空数组[]）：
[{{"key": "记忆键名", "value": "具体内容"}}]

只返回 JSON，不要任何其他文字。"""
    try:
        from langchain_core.messages import HumanMessage
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        text = response.content.strip().strip("```json").strip("```").strip()
        items = json.loads(text)
        count = 0
        for item in items:
            if isinstance(item, dict) and "key" in item and "value" in item:
                save_memory(item["key"], item["value"], source="auto")
                count += 1
        if count:
            logger.info(f"自动提取了 {count} 条长期记忆")
        return count
    except Exception as e:
        logger.debug(f"记忆提取失败（非关键）: {e}")
        return 0


async def generate_session_title(messages: list, llm) -> str:
    """
    根据对话内容生成简短标题（8字以内）。
    在后台异步调用，不影响主流程。
    """
    if not messages:
        return ""
    # 只取前几条消息就够了
    snippet = "\n".join(
        f"{m['role']}: {m['content'][:200]}"
        for m in messages[:6]
        if m['role'] in ('human', 'user', 'assistant', 'ai')
    )
    prompt = f"""根据以下对话内容，生成一个简短的标题（不超过10个字，直接输出标题，不要任何标点或解释）：

{snippet}"""
    try:
        from langchain_core.messages import HumanMessage
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        title = response.content.strip().strip("\"'").strip()[:20]
        return title
    except Exception as e:
        logger.debug(f"标题生成失败: {e}")
        return ""


# ──────────────────────────────────────────
# Agent 可调用的记忆工具
# ──────────────────────────────────────────

from langchain_core.tools import tool


@tool
def memory_save(key: str, value: str) -> str:
    """
    保存一条长期记忆。key 是记忆的唯一标识（简短英文或中文），value 是具体内容。
    例如：memory_save(key="用户编程语言偏好", value="偏好使用Python，不喜欢Java")
    同一个 key 再次保存会覆盖旧值。
    """
    try:
        save_memory(key, value, source="agent")
        return f"已保存记忆：{key} = {value}"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def memory_delete(key: str) -> str:
    """
    删除一条长期记忆。
    例如：memory_delete(key="用户编程语言偏好")
    """
    try:
        delete_memory(key)
        return f"已删除记忆：{key}"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def memory_list() -> str:
    """
    列出所有长期记忆。当需要了解已记住的信息时调用。
    """
    memories = load_all_memories()
    if not memories:
        return "当前没有任何长期记忆。"
    lines = [f"- {m['key']}: {m['value']} (来源: {m['source'] or '未知'})" for m in memories]
    return "当前长期记忆：\n" + "\n".join(lines)


MEMORY_TOOLS = [memory_save, memory_delete, memory_list]
