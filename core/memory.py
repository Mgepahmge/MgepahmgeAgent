"""
memory.py - 长期记忆管理
负责从对话中提取记忆，并在每次对话时注入 system prompt。
"""
from __future__ import annotations
import logging
import time
from .database import load_all_memories, save_memory, delete_memory

logger = logging.getLogger(__name__)

# 超过这么多条消息后，触发记忆提取
MEMORY_EXTRACT_THRESHOLD = 10


def build_memory_prompt() -> str:
    """构建注入 system prompt 的记忆片段"""
    memories = load_all_memories()
    if not memories:
        return ""
    lines = [f"- {m['key']}: {m['value']}" for m in memories]
    return "【长期记忆】\n" + "\n".join(lines)


async def extract_memories(messages: list, llm) -> int:
    """
    让 LLM 从对话中提取值得长期记住的信息，保存到数据库。
    返回提取的记忆条数。
    """
    if not messages:
        return 0

    conversation = "\n".join(
        f"{m['role']}: {m['content'][:500]}"
        for m in messages[-20:]  # 只看最近20条
        if m['role'] in ('human', 'user', 'assistant', 'ai')
    )

    prompt = f"""分析以下对话，提取值得长期记住的用户信息（偏好、习惯、重要事实等）。
只提取确定的、有价值的信息，不要猜测。每条记忆用 JSON 格式返回。

对话内容：
{conversation}

返回格式（JSON 数组，如无有价值信息则返回空数组[]）：
[{{"key": "记忆键名", "value": "具体内容"}}]

只返回 JSON，不要任何其他文字。"""

    try:
        from langchain_core.messages import HumanMessage
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        text = response.content.strip()
        # 清理可能的 markdown 代码块
        text = text.strip("```json").strip("```").strip()
        import json
        items = json.loads(text)
        count = 0
        for item in items:
            if isinstance(item, dict) and "key" in item and "value" in item:
                save_memory(item["key"], item["value"], source="auto")
                count += 1
        if count:
            logger.info(f"提取了 {count} 条长期记忆")
        return count
    except Exception as e:
        logger.debug(f"记忆提取失败（非关键）: {e}")
        return 0
