"""
agent_graph.py - LangGraph Agent 核心
支持任意 Provider（Anthropic / OpenAI 兼容 / Ollama）
"""
from __future__ import annotations
import logging
from typing import Annotated, Optional
import operator

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个运行在 Ubuntu 系统上的全能 AI 助手，拥有以下能力：

【文件系统】读写 workspace 内的文件，列目录，查找文件
【Web】搜索互联网，抓取网页内容
【Shell】在 workspace 中执行白名单命令（git, python, grep, nvidia-smi 等）
【MCP 工具】通过 MCP 协议连接的外部服务（按实际加载的工具为准）
【知识库】如果启用了 RAG，可从私有知识库中检索相关内容

工作原则：
- 收到任务后，先思考最优路径，再一步步执行
- 需要信息时主动使用工具获取，不要凭空猜测
- 执行有破坏性操作（删除、覆盖）前，先说明再执行
- 代码和命令执行后，检查输出是否符合预期
- 完成后给出清晰的结果摘要

当前工作目录：{workspace}
"""


# ──────────────────────────────────────────
# LLM 工厂：根据 ProviderProfile 创建对应 LangChain LLM
# ──────────────────────────────────────────

def build_llm(profile):
    """
    根据 ProviderProfile.type 创建对应的 LangChain chat model。

    type = "anthropic" → ChatAnthropic
    type = "openai"    → ChatOpenAI（也支持任意 OpenAI 兼容接口，如 DeepSeek、Qwen、本地 vLLM）
    type = "ollama"    → ChatOllama
    """
    ptype = profile.type.lower()

    if ptype == "anthropic":
        from langchain_anthropic import ChatAnthropic
        kwargs = dict(
            model=profile.model,
            api_key=profile.api_key,
            max_tokens=profile.max_tokens,
        )
        if profile.base_url:
            kwargs["base_url"] = profile.base_url
        llm = ChatAnthropic(**kwargs)
        logger.info(f"Provider: Anthropic | 模型: {profile.model}")

    elif ptype == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = dict(
            model=profile.model,
            api_key=profile.api_key or "placeholder",  # 部分兼容接口不需要 key
            max_tokens=profile.max_tokens,
        )
        if profile.base_url:
            kwargs["base_url"] = profile.base_url
        llm = ChatOpenAI(**kwargs)
        logger.info(f"Provider: OpenAI 兼容 ({profile.base_url or 'openai'}) | 模型: {profile.model}")

    elif ptype == "ollama":
        from langchain_ollama import ChatOllama
        kwargs = dict(model=profile.model)
        if profile.base_url:
            kwargs["base_url"] = profile.base_url
        llm = ChatOllama(**kwargs)
        logger.info(f"Provider: Ollama ({profile.base_url}) | 模型: {profile.model}")

    else:
        raise ValueError(
            f"未知 provider type: '{ptype}'，支持: anthropic / openai / ollama"
        )

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

def make_retrieve_node(kb):
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
        results = kb.search(query, k=4)
        if not results:
            return {"rag_context": ""}
        ctx = "\n\n".join(
            f"[来源: {r['source']} | 相关度: {r['score']}]\n{r['content']}"
            for r in results
        )
        logger.debug(f"RAG 检索到 {len(results)} 条结果")
        return {"rag_context": ctx}
    return retrieve


def make_llm_node(llm_with_tools, workspace: str):
    def call_llm(state: AgentState) -> dict:
        rag_ctx = state.get("rag_context", "")
        system_content = SYSTEM_PROMPT.format(workspace=workspace)
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

def build_agent(cfg, kb, all_tools: list):
    """
    cfg: AgentConfig
    kb:  KnowledgeBase | None
    all_tools: 内置工具 + MCP 工具
    """
    profile = cfg.providers.active()
    if profile is None:
        raise RuntimeError(
            "没有激活的 LLM Provider，请用 /provider add 添加一个，或编辑 config/providers.json"
        )

    llm = build_llm(profile)
    llm_with_tools = llm.bind_tools(all_tools)

    retrieve_node = make_retrieve_node(kb)
    llm_node = make_llm_node(llm_with_tools, cfg.workspace_dir)
    tool_node = ToolNode(all_tools)

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("llm", llm_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "llm")
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
