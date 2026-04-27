"""
agent_registry.py - 多 Agent 注册表和运行时管理

每个 Agent 有独立的：
  - LangGraph 编译图
  - LLM 实例
  - 工具集（base_tools + skills 合并）
  - 专属事件循环线程（实现并发）
  - 对话状态（InMemorySaver）

Agent 配置从 agents/*.yaml 加载。
"""
from __future__ import annotations
import asyncio
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)

AGENTS_DIR = Path(__file__).parent.parent / "agents"


# ──────────────────────────────────────────
# Agent 配置数据结构
# ──────────────────────────────────────────

@dataclass
class MemoryConfig:
    use_global: bool = True
    use_private: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryConfig":
        return cls(
            use_global=bool(d.get("global", True)),
            use_private=bool(d.get("private", True)),
        )


@dataclass
class AgentProfile:
    """
    一个 Agent 的完整配置。
    id       : 文件名（不含扩展名），全局唯一
    name     : 显示名称
    provider : LLM Provider 名称，空字符串表示使用全局激活的 Provider
    base_tools : 可用工具名称列表，空列表表示使用全部工具
    skills   : 激活的 Skill ID 列表
    workdir  : 工作目录，空字符串表示使用全局 WORKSPACE_DIR
    memory   : 记忆配置
    """
    id: str
    name: str
    description: str = ""
    provider: str = ""
    system_prompt: str = ""
    workdir: str = ""
    base_tools: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    @classmethod
    def from_file(cls, path: Path) -> "AgentProfile":
        if yaml is None:
            raise ImportError("PyYAML 未安装，请运行：pip install pyyaml")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        aid = path.stem
        name = str(data.get("name", "")).strip()
        if not name:
            raise ValueError(f"Agent '{aid}' 缺少必填字段 'name'")
        mem_data = data.get("memory", {})
        if not isinstance(mem_data, dict):
            mem_data = {}
        return cls(
            id=aid,
            name=name,
            description=str(data.get("description", "")).strip(),
            provider=str(data.get("provider", "")).strip(),
            system_prompt=str(data.get("system_prompt", "")).strip(),
            workdir=str(data.get("workdir", "")).strip(),
            base_tools=[str(t).strip() for t in (data.get("base_tools") or [])],
            skills=[str(s).strip() for s in (data.get("skills") or [])],
            memory=MemoryConfig.from_dict(mem_data),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "provider": self.provider,
            "system_prompt": self.system_prompt,
            "workdir": self.workdir,
            "base_tools": self.base_tools,
            "skills": self.skills,
            "memory": {
                "global": self.memory.use_global,
                "private": self.memory.use_private,
            },
        }


# ──────────────────────────────────────────
# Agent 运行时实例
# ──────────────────────────────────────────

@dataclass
class AgentRuntime:
    """
    一个运行中的 Agent 实例。
    每个 Agent 有独立的事件循环线程，实现真正的并发。
    """
    profile: AgentProfile
    graph: object        # 编译好的 LangGraph
    llm: object          # LLM 实例
    tools: list          # 合并后的工具列表
    knowledge_ids: list[str]  # 合并后的知识集合 ID

    _loop: asyncio.AbstractEventLoop = field(default=None, repr=False)
    _thread: threading.Thread = field(default=None, repr=False)
    _restored_sessions: set = field(default_factory=set, repr=False)

    def __post_init__(self):
        # 每个 Agent 独立的持久化事件循环
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"agent-{self.profile.id}",
            daemon=True,
        )
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def invoke(self, query: str, session_id: str) -> str:
        """同步调用 Agent，阻塞直到返回结果"""
        from langchain_core.messages import HumanMessage
        future = asyncio.run_coroutine_threadsafe(
            self._ainvoke(query, session_id), self._loop
        )
        return future.result()

    async def _ainvoke(self, query: str, session_id: str) -> str:
        from langchain_core.messages import HumanMessage, AIMessage
        from core.database import load_messages

        # 首次使用该 session 时从数据库恢复历史上下文
        if session_id not in self._restored_sessions:
            history = load_messages(session_id)
            prior = history[:-1]  # 不含刚存入的当前消息
            if prior:
                prior_msgs = []
                for m in prior:
                    if m["role"] in ("human", "user"):
                        prior_msgs.append(HumanMessage(content=m["content"]))
                    elif m["role"] in ("assistant", "ai"):
                        prior_msgs.append(AIMessage(content=m["content"]))
                if prior_msgs:
                    await self.graph.ainvoke(
                        {"messages": prior_msgs,
                         "workspace": self.profile.workdir or ""},
                        config={"configurable": {"thread_id": session_id}},
                    )
            self._restored_sessions.add(session_id)

        result = await self.graph.ainvoke(
            {"messages": [HumanMessage(content=query)],
             "workspace": self.profile.workdir or ""},
            config={"configurable": {"thread_id": session_id}},
        )
        last = result["messages"][-1]
        return last.content if hasattr(last, "content") else str(last)

    def stop(self):
        """停止 Agent 的事件循环"""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)


# ──────────────────────────────────────────
# Agent 注册表
# ──────────────────────────────────────────

class AgentRegistry:
    """
    管理所有 Agent 配置和运行时实例。

    接口：
      registry.profiles()          → 所有 AgentProfile 列表
      registry.get_profile(aid)    → AgentProfile | None
      registry.get_runtime(aid)    → AgentRuntime | None
      registry.start(aid, cfg, kb, all_tools)  → 启动 Agent
      registry.stop(aid)           → 停止 Agent
      registry.reload_profiles()   → 重新扫描 agents/ 目录
    """

    def __init__(self):
        self._profiles: dict[str, AgentProfile] = {}
        self._runtimes: dict[str, AgentRuntime] = {}
        self.reload_profiles()

    def reload_profiles(self):
        """扫描 agents/ 目录，加载所有 .yaml 文件"""
        self._profiles.clear()
        if not AGENTS_DIR.exists():
            AGENTS_DIR.mkdir(parents=True)
            logger.info("已创建 agents/ 目录")
            return
        for path in sorted(AGENTS_DIR.glob("*.yaml")):
            if path.stem.startswith("_"):
                continue
            try:
                profile = AgentProfile.from_file(path)
                self._profiles[profile.id] = profile
                logger.info(f"Agent [{profile.id}]: {profile.name} 配置加载成功")
            except Exception as e:
                logger.warning(f"Agent [{path.stem}] 配置加载失败: {e}")
        logger.info(f"共加载 {len(self._profiles)} 个 Agent 配置")

    def profiles(self) -> list[AgentProfile]:
        return list(self._profiles.values())

    def get_profile(self, aid: str) -> Optional[AgentProfile]:
        return self._profiles.get(aid)

    def get_runtime(self, aid: str) -> Optional[AgentRuntime]:
        return self._runtimes.get(aid)

    def is_running(self, aid: str) -> bool:
        return aid in self._runtimes

    def start(self, aid: str, global_cfg, kb, all_tools: list) -> AgentRuntime:
        """
        启动一个 Agent，构建其独立的 LangGraph 图和工具集。
        如果已在运行则直接返回现有实例。
        """
        if aid in self._runtimes:
            return self._runtimes[aid]

        profile = self._profiles.get(aid)
        if profile is None:
            raise ValueError(f"Agent '{aid}' 配置不存在")

        # 确定 Provider
        from core.config import ProviderRegistry
        providers: ProviderRegistry = global_cfg.providers
        if profile.provider:
            provider_profile = providers.get(profile.provider)
            if provider_profile is None:
                raise ValueError(
                    f"Agent '{aid}' 指定的 Provider '{profile.provider}' 不存在"
                )
        else:
            provider_profile = providers.active()
            if provider_profile is None:
                raise ValueError("没有激活的 LLM Provider")

        # 过滤工具：base_tools 为空则使用全部
        # 支持工具名、插件名（source）、mcp:server 三种引用方式
        from core.skill_loader import resolve_tool_refs
        if profile.base_tools:
            filtered_tools = resolve_tool_refs(profile.base_tools, all_tools)
            if not filtered_tools:
                logger.warning(f"Agent '{aid}' 的 base_tools 未匹配任何工具，回退到全部工具")
                filtered_tools = list(all_tools)
        else:
            filtered_tools = list(all_tools)

        # 合并 Skill
        from core.skill_loader import skill_registry, merge_skills
        if profile.skills:
            skill_prompt, merged_tools, knowledge_ids = merge_skills(
                profile.skills, skill_registry, filtered_tools
            )
        else:
            skill_prompt, merged_tools, knowledge_ids = "", filtered_tools, []

        # 确定工作目录
        workdir = profile.workdir or global_cfg.workspace_dir

        # 构建 Agent 专属 system prompt
        agent_system_prompt = profile.system_prompt
        full_skill_prompt = skill_prompt

        # 构建 LangGraph
        from core.agent_graph import build_llm, make_retrieve_node, make_llm_node
        from core.agent_graph import AgentState, should_continue, SYSTEM_PROMPT
        from langgraph.graph import StateGraph, END
        from langgraph.prebuilt import ToolNode
        from langgraph.checkpoint.memory import InMemorySaver
        from core.database import init_db
        init_db()

        llm = build_llm(provider_profile)
        llm_with_tools = llm.bind_tools(merged_tools)

        # 为此 Agent 构建专属的 LLM 节点（注入专属 prompt）
        def make_agent_llm_node(llm_wt, wdir, agent_id, sp, skp, use_global_mem, use_private_mem):
            def call_llm(state: AgentState) -> dict:
                from core.memory import build_memory_prompt
                from langchain_core.messages import SystemMessage
                rag_ctx = state.get("rag_context", "")

                # 根据记忆配置决定注入哪些记忆
                if use_global_mem and use_private_mem:
                    mem_agent_id = agent_id
                elif use_global_mem:
                    mem_agent_id = ""   # 只加载全局
                elif use_private_mem:
                    mem_agent_id = agent_id  # 只加载私有（通过过滤实现）
                else:
                    mem_agent_id = None

                memory_section = build_memory_prompt(
                    agent_id=mem_agent_id if mem_agent_id is not None else ""
                ) if mem_agent_id is not None else ""

                system_content = SYSTEM_PROMPT.format(
                    workspace=wdir,
                    memory_section=memory_section,
                )
                if sp:
                    system_content += f"\n\n{sp}"
                if skp:
                    system_content += f"\n\n{skp}"
                if rag_ctx:
                    system_content += f"\n\n【知识库检索结果】\n{rag_ctx}"

                messages = [SystemMessage(content=system_content)] + state["messages"]
                response = llm_wt.invoke(messages)
                return {"messages": [response]}
            return call_llm

        retrieve_node = make_retrieve_node(kb, knowledge_ids or None)
        llm_node = make_agent_llm_node(
            llm_with_tools, workdir, profile.id,
            agent_system_prompt, full_skill_prompt,
            profile.memory.use_global, profile.memory.use_private,
        )
        tool_node = ToolNode(merged_tools)

        graph = StateGraph(AgentState)
        graph.add_node("retrieve", retrieve_node)
        graph.add_node("llm", llm_node)
        graph.add_node("tools", tool_node)
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "llm")
        graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
        graph.add_edge("tools", "llm")
        compiled = graph.compile(checkpointer=InMemorySaver())

        runtime = AgentRuntime(
            profile=profile,
            graph=compiled,
            llm=llm,
            tools=merged_tools,
            knowledge_ids=knowledge_ids,
        )
        self._runtimes[aid] = runtime
        logger.info(
            f"Agent [{aid}] 已启动: provider={provider_profile.name}, "
            f"tools={len(merged_tools)}, skills={profile.skills}, workdir={workdir}"
        )
        return runtime

    def stop(self, aid: str) -> bool:
        runtime = self._runtimes.pop(aid, None)
        if runtime is None:
            return False
        runtime.stop()
        logger.info(f"Agent [{aid}] 已停止")
        return True

    def stop_all(self):
        for aid in list(self._runtimes.keys()):
            self.stop(aid)


# 全局单例
agent_registry = AgentRegistry()
