"""
memory.py - 长期记忆核心逻辑
负责：记忆注入 system prompt、自动提取（可开关）、对话标题生成
注意：Agent 可调用的记忆工具在 tools/memory_tools.py（插件）
"""
from __future__ import annotations
import json
import logging
import os
from .database import load_all_memories, save_memory

logger = logging.getLogger(__name__)

# 每隔多少条消息触发一次自动提取
MEMORY_EXTRACT_THRESHOLD = 10

# 自动提取开关，默认关闭，在 .env 中设置 AUTO_EXTRACT_MEMORY=true 开启
AUTO_EXTRACT_MEMORY = os.getenv("AUTO_EXTRACT_MEMORY", "false").lower() == "true"


def build_memory_prompt() -> str:
    """构建注入 system prompt 的记忆片段"""
    memories = load_all_memories()
    if not memories:
        return ""
    lines = [f"- {m['key']}: {m['value']}" for m in memories]
    return "【长期记忆】\n" + "\n".join(lines)


async def extract_memories(messages: list, llm) -> int:
    """
    从对话中自动提取长期记忆。
    仅在 AUTO_EXTRACT_MEMORY=true 时执行，默认关闭。
    """
    if not AUTO_EXTRACT_MEMORY:
        return 0
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
    """根据对话内容生成简短标题（不超过10字）"""
    if not messages:
        return ""
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
