"""
agent_graph.py - LangGraph Agent 核心
- 支持多 Provider
- 集成长期记忆注入
- 使用 AsyncSqliteSaver 持久化对话（支持 ainvoke）
"""
from __future__ import annotations
import logging
from typing import Annotated
import operator
import os
from pathlib import Path

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

DB_PATH = Path(os.getenv("AGENT_DB", "/root/agent/data/agent.db"))

SYSTEM_PROMPT = """你是一个运行在 Ubuntu 系统上的全能 AI 助手，拥有操作系统级别的完整能力：

【文件操作】通过 MCP filesystem 工具读写、编辑、移动、搜索系统内任意有权限的文件和目录
【Web】使用 web_search 搜索互联网，fetch_url 抓取网页内容
【Shell】使用 run_shell 执行任意 shell 命令，安装软件，运行脚本，管理进程
【其他 MCP 工具】通过 MCP 协议连接的外部服务（git、数据库等）
【知识库】如果启用了 RAG，可从私有知识库中检索相关内容
【长期记忆】使用 memory_save/memory_delete/memory_list 工具主动管理记忆
  - 当用户要求记住某件事时，调用 memory_save 保存
  - 当用户要求忘记某件事时，调用 memory_delete 删除
  - 记忆会在每次对话时自动注入到上下文中

工作原则：
- 收到任务后先思考最优路径，再逐步执行
- 需要信息时主动使用工具获取，不要凭空猜测
- 【重要】执行以下破坏性操作前，必须先用文字清晰说明：
    * 删除文件或目录
    * 覆盖已有文件内容
    * 修改系统配置文件
    * 卸载软件包
    * 任何不可逆的操作
  说明后等待用户明确确认（回复"确认"或"yes"等），再执行。
- 代码和命令执行后检查输出是否符合预期
- 完成后给出清晰的结果摘要

workspace 目录（{workspace}）是与宿主机交换文件的接口。
{memory_section}"""


# ──────────────────────────────────────────
# LLM 工厂
# ──────────────────────────────────────────

def build_llm(profile):
    ptype = profile.type.lower()
    key = profile.resolved_api_key

    if ptype == "anthropic":
        from langchain_anthropic import ChatAnthropic
        kwargs = dict(model=profile.model, api_key=key, max_tokens=profile.max_tokens)
        if profile.base_url:
            kwargs["base_url"] = profile.base_url
        llm = ChatAnthropic(**kwargs)

    elif ptype == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = dict(
            model=profile.model,
            api_key=key or "placeholder",
            max_tokens=profile.max_tokens,
        )
        if profile.base_url:
            kwargs["base_url"] = profile.base_url
        llm = ChatOpenAI(**kwargs)

    elif ptype == "ollama":
        from langchain_ollama import ChatOllama
        kwargs = dict(model=profile.model)
        if profile.base_url:
            kwargs["base_url"] = profile.base_url
        llm = ChatOllama(**kwargs)

    else:
        raise ValueError(f"未知 provider type: '{ptype}'，支持: anthropic / openai / ollama")

    logger.info(f"Provider: {profile.name} ({profile.type}) | 模型: {profile.model}")
    return llm


# ──────────────────────────────────────────
# Graph State
# ──────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    rag_context: str
    workspace: str


# ──────────────────────────────────────────
# 节点函数
# ──────────────────────────────────────────

def make_retrieve_node(kb, knowledge_ids: list[str] | None = None):
    """
    knowledge_ids: Skill 指定的知识集合 ID 列表，None 表示检索全库
    """
    def retrieve(state: AgentState) -> dict:
        if kb is None:
            return {"rag_context": ""}
        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        if not last_human:
            return {"rag_context": ""}
        query = last_human.content if isinstance(last_human.content, str) else ""
        results = kb.search(query, k=4, collection_ids=knowledge_ids or None)
        if not results:
            return {"rag_context": ""}
        ctx = "\n\n".join(
            f"[来源: {r['source']} | 相关度: {r['score']}]\n{r['content']}"
            for r in results
        )
        logger.debug(f"RAG 检索到 {len(results)} 条结果")
        return {"rag_context": ctx}
    return retrieve


def make_llm_node(llm_with_tools, workspace: str, skill_prompt: str = ""):
    """
    skill_prompt: 所有激活 Skill 的 system_prompt 拼接片段
    """
    def call_llm(state: AgentState) -> dict:
        from core.memory import build_memory_prompt
        rag_ctx = state.get("rag_context", "")
        memory_section = build_memory_prompt(agent_id="")
        system_content = SYSTEM_PROMPT.format(
            workspace=workspace,
            memory_section=memory_section,
        )
        # 追加 Skill 提示片段
        if skill_prompt:
            system_content += f"\n\n{skill_prompt}"
        if rag_ctx:
            system_content += f"\n\n【知识库检索结果】\n{rag_ctx}"
        messages = [SystemMessage(content=system_content)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    return call_llm


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


# ──────────────────────────────────────────
# 图组装
# ──────────────────────────────────────────

def build_agent(cfg, kb, all_tools: list, skill_ids: list[str] | None = None):
    """
    skill_ids: 激活的 Skill ID 列表，None 表示不使用任何 Skill
    """
    from core.database import init_db
    from core.skill_loader import skill_registry, merge_skills
    init_db()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    profile = cfg.providers.active()
    if profile is None:
        raise RuntimeError(
            "没有激活的 LLM Provider，请用 /provider add 添加，或编辑 config/providers.json"
        )

    # 合并 Skill：追加 prompt、扩展工具、过滤知识
    if skill_ids:
        skill_prompt, merged_tools, knowledge_ids = merge_skills(
            skill_ids, skill_registry, all_tools
        )
        logger.info(f"激活 Skill: {skill_ids}")
    else:
        skill_prompt, merged_tools, knowledge_ids = "", list(all_tools), []

    llm = build_llm(profile)
    llm_with_tools = llm.bind_tools(merged_tools)

    retrieve_node = make_retrieve_node(kb, knowledge_ids or None)
    llm_node = make_llm_node(llm_with_tools, cfg.workspace_dir, skill_prompt)
    tool_node = ToolNode(merged_tools)

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("llm", llm_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "llm")
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")

    # InMemorySaver：进程内持久化，支持 ainvoke，无需额外依赖。
    # 对话消息的长期持久化由 core/database.py 的 messages 表负责，
    # 重启后通过 /session load 从数据库恢复上下文。
    from langgraph.checkpoint.memory import InMemorySaver
    checkpointer = InMemorySaver()
    logger.info("使用 InMemorySaver（对话消息持久化由 database.py 负责）")

    return graph.compile(checkpointer=checkpointer), llm
