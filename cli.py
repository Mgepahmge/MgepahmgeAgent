"""
cli.py - 命令行入口
支持：多 Provider、持久化对话、长期记忆、后台任务、工具插件
"""
from __future__ import annotations
import sys
import uuid
import logging
import asyncio
import click
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML

console = Console()

_agent = None      # 兼容旧代码，指向当前 Agent 的 AgentRuntime
_llm = None
_kb = None
_cfg = None
_all_tools = []
_current_session_id = None
_restored_sessions: set = set()
_active_skills: list[str] = []
_current_agent_id: str = "default"   # 当前交互的 Agent ID


def _load_active_skills() -> list[str]:
    """从 config/active_skills.json 读取激活的 Skill ID 列表"""
    import json
    path = Path(__file__).parent / "config" / "active_skills.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def _save_active_skills(skill_ids: list[str]):
    """持久化激活的 Skill ID 列表"""
    import json
    path = Path(__file__).parent / "config" / "active_skills.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(skill_ids, ensure_ascii=False, indent=2))


def _rebuild_agent_with_skills():
    """Skill 变更后重建 Agent（保留现有工具和 KB）"""
    import logging
    from core.agent_graph import build_agent
    global _agent
    _agent, _ = build_agent(_cfg, _kb, _all_tools, skill_ids=_active_skills)
    logging.getLogger(__name__).info(f"Agent 已重建，激活 Skill: {_active_skills}")


def _bootstrap():
    global _agent, _llm, _kb, _cfg, _all_tools

    sys.path.insert(0, str(Path(__file__).parent))
    from core.config import config
    _cfg = config

    log_path = Path(config.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 启动阶段：同时输出到终端和文件
    _stderr_handler = logging.StreamHandler(sys.stderr)
    _stderr_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            _stderr_handler,
        ],
    )
    for noisy in ["httpx", "httpcore", "anthropic", "urllib3", "openai"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    with console.status("[bold cyan]初始化组件...", spinner="dots"):
        # 初始化数据库
        from core.database import init_db
        init_db()

        # RAG
        kb = None
        if config.rag.enabled:
            from rag.knowledge_base import KnowledgeBase
            kb = KnowledgeBase.connect(config.rag)
        _kb = kb

        # 内置工具
        from tools.plugin_loader import load_builtin_tools
        all_tools = load_builtin_tools(config.workspace_dir)

        # 插件工具（自动发现）
        from tools.plugin_loader import discover_plugins
        plugin_tools = discover_plugins()
        all_tools.extend(plugin_tools)

        # MCP 工具
        if config.mcp.servers:
            from tools.mcp_loader import load_mcp_tools_sync
            all_tools.extend(load_mcp_tools_sync(config.mcp))

        _all_tools = all_tools

        # 构建 Agent
        from core.agent_graph import build_agent
        result = build_agent(config, kb, all_tools)
        _agent, _llm = result

    # 启动完成：移除终端日志输出，后续日志静默写入文件
    logging.getLogger().removeHandler(_stderr_handler)

    profile = config.providers.active()
    from core.agent_registry import agent_registry
    runtime = agent_registry.get_runtime(_current_agent_id)
    agent_profile = agent_registry.get_profile(_current_agent_id)
    provider_name = (agent_profile.provider or config.providers.active_name()) if agent_profile else config.providers.active_name()
    workdir_display = (agent_profile.workdir or config.workspace_dir) if agent_profile else config.workspace_dir
    skills_display = ", ".join(agent_profile.skills) if agent_profile and agent_profile.skills else "[dim]无[/]"
    console.print(Panel(
        f"[bold green]Agent 就绪[/]\n"
        f"当前 Agent: [cyan]{_current_agent_id}[/] "
        f"([dim]{agent_profile.name if agent_profile else '未知'}[/])\n"
        f"Provider  : [cyan]{provider_name}[/] "
        f"([dim]{profile.type} / {profile.model}[/])\n"
        f"工作目录  : [cyan]{workdir_display}[/]\n"
        f"RAG 知识库: {'[green]已连接[/]' if kb else '[yellow]未启用[/]'}\n"
        f"工具数量  : [cyan]{len(runtime.tools) if runtime else len(all_tools)}[/]\n"
        f"激活 Skill: {skills_display}",
        title="🤖 CLI Agent",
        border_style="cyan",
    ))
    return all_tools


def _rebuild_agent():
    global _agent, _llm
    from core.agent_graph import build_agent
    _agent, _llm = build_agent(_cfg, _kb, _all_tools)


def _run_query(query: str, thread_id: str) -> str:
    from core.database import save_message
    from core.agent_registry import agent_registry

    # 保存用户消息
    save_message(thread_id, "human", query)

    # 通过当前 Agent 的 Runtime 调用（每个 Agent 有独立事件循环）
    runtime = agent_registry.get_runtime(_current_agent_id)
    if runtime is None:
        return "当前 Agent 未启动，请检查配置"

    answer = runtime.invoke(query, thread_id)

    # 保存 AI 回复
    save_message(thread_id, "assistant", answer)

    from core.database import count_messages, load_messages, update_session_summary
    msg_count = count_messages(thread_id)

    # 第2条消息后（user+assistant各1条）异步生成对话标题
    if msg_count == 2:
        async def _gen_title():
            from core.memory import generate_session_title
            msgs = load_messages(thread_id)
            title = await generate_session_title(msgs, _llm)
            if title:
                from core.database import get_conn
                with get_conn() as conn:
                    conn.execute("UPDATE sessions SET name=? WHERE id=?", (title, thread_id))
        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_gen_title())
            loop.close()
        except Exception:
            pass

    # 每 N 条消息自动提取长期记忆
    from core.memory import MEMORY_EXTRACT_THRESHOLD, extract_memories
    if msg_count % MEMORY_EXTRACT_THRESHOLD == 0:
        async def _extract():
            msgs = load_messages(thread_id)
            await extract_memories(msgs, _llm)
        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_extract())
            loop.close()
        except Exception:
            pass

    return answer


# ──────────────────────────────────────────
# /session 指令
# ──────────────────────────────────────────

def _session_list() -> list:
    """列出对话，返回 sessions 列表供索引使用"""
    from core.database import list_sessions
    sessions = list_sessions(agent_id=_current_agent_id)
    if not sessions:
        console.print("[dim]暂无对话记录[/]")
        return []
    import time
    table = Table(title="对话记录", border_style="cyan")
    table.add_column("#", style="bold cyan", width=4)
    table.add_column("名称")
    table.add_column("最后活跃")
    table.add_column("摘要", max_width=40)
    for i, s in enumerate(sessions, 1):
        ts = time.strftime("%m-%d %H:%M", time.localtime(s["updated_at"]))
        table.add_row(str(i), s["name"], ts, s["summary"] or "[dim]-[/]")
    console.print(table)
    return sessions


def _session_new(name: str = "") -> str:
    from core.database import create_session
    sid = create_session(name, agent_id=_current_agent_id)
    console.print(f"[green]✓ 新建对话 {sid[:8]}[/]")
    return sid


def _show_session_history(sid: str):
    """展示一个对话的历史消息"""
    import time
    from core.database import load_messages, get_session
    session = get_session(sid)
    messages = load_messages(sid)
    if not messages:
        console.print("[dim]该对话暂无消息记录[/]")
        return
    name = session["name"] if session else sid[:8]
    console.print(f"\n[dim]── 对话历史：{name} ──[/]")
    for m in messages:
        if m["role"] in ("human", "user"):
            console.print(f"[bold blue]你[/]  {m['content'][:500]}")
        elif m["role"] in ("assistant", "ai"):
            console.print(f"[bold green]Agent[/]  {m['content'][:500]}")
    console.print("[dim]── 历史结束 ──[/]\n")


def _session_load(identifier: str) -> str | None:
    """支持序号（数字）或 ID 前缀两种方式"""
    from core.database import list_sessions
    sessions = list_sessions(100, agent_id=_current_agent_id)
    target = None
    if identifier.isdigit():
        idx = int(identifier) - 1
        if 0 <= idx < len(sessions):
            target = sessions[idx]
        else:
            console.print(f"[red]序号 {identifier} 超出范围（共 {len(sessions)} 个对话）[/]")
            return None
    else:
        for s in sessions:
            if s["id"].startswith(identifier):
                target = s
                break
    if target is None:
        console.print(f"[red]找不到对话[/]")
        return None
    console.print(f"[green]✓ 已加载对话: {target['name']}[/]")
    if Confirm.ask("显示该对话的历史记录？", default=False):
        _show_session_history(target["id"])
    return target["id"]


def _session_delete(identifier: str):
    """支持序号（数字）或 ID 前缀两种方式，和 _session_load 逻辑一致"""
    from core.database import list_sessions, delete_session
    sessions = list_sessions(100, agent_id=_current_agent_id)
    target = None
    if identifier.isdigit():
        idx = int(identifier) - 1
        if 0 <= idx < len(sessions):
            target = sessions[idx]
        else:
            console.print(f"[red]序号 {identifier} 超出范围（共 {len(sessions)} 个对话）[/]")
            return
    else:
        for s in sessions:
            if s["id"].startswith(identifier):
                target = s
                break
    if target is None:
        console.print(f"[red]找不到对话[/]")
        return
    if Confirm.ask(f"确认删除对话 '{target['name']}'？", default=False):
        delete_session(target["id"])
        console.print(f"[green]✓ 已删除[/]")


def _handle_session_cmd(parts: list[str], current_sid: str) -> str:
    sub = parts[1] if len(parts) > 1 else "list"
    if sub in ("list", "ls"):
        _session_list()  # 仅展示，不切换
        return current_sid
    elif sub == "new":
        name = " ".join(parts[2:]) if len(parts) > 2 else ""
        return _session_new(name)
    elif sub in ("load", "switch") and len(parts) > 2:
        new_sid = _session_load(parts[2])
        return new_sid or current_sid
    elif sub in ("delete", "rm") and len(parts) > 2:
        _session_delete(parts[2])
        return current_sid
    else:
        console.print(Panel(
            "/session list              列出所有对话\n"
            "/session new [名称]        新建对话\n"
            "/session load <序号/ID>    加载已有对话\n"
            "/session delete <序号/ID>  删除对话",
            title="session 指令",
        ))
        return current_sid


# ──────────────────────────────────────────
# /memory 指令
# ──────────────────────────────────────────

def _memory_list(scope: str = "all"):
    """
    列出长期记忆。
    scope='all'    : 全局 + Agent 私有（默认）
    scope='global' : 仅全局
    scope='agent'  : 仅 Agent 私有
    """
    import time
    from core.database import load_memories, load_all_memories

    if scope == "global":
        memories = load_memories(scope="global", agent_id="")
        title = "全局长期记忆"
    elif scope == "agent":
        memories = load_memories(scope="agent", agent_id="")
        title = "Agent 私有记忆"
    else:
        memories = load_all_memories(agent_id="")
        title = "长期记忆（全局 + 私有）"

    if not memories:
        console.print(f"[dim]暂无{title}[/]")
        return

    table = Table(title=title, border_style="cyan")
    table.add_column("#", width=4, style="bold cyan")
    table.add_column("类型", width=6)
    table.add_column("键", style="bold")
    table.add_column("内容")
    table.add_column("来源", style="dim", width=8)
    table.add_column("更新时间", style="dim", width=12)

    for i, m in enumerate(memories, 1):
        scope_label = "[cyan]全局[/]" if m["scope"] == "global" else "[yellow]私有[/]"
        ts = time.strftime("%m-%d %H:%M", time.localtime(m["updated_at"]))
        table.add_row(str(i), scope_label, m["key"], m["value"],
                      m["source"] or "-", ts)
    console.print(table)


def _memory_set(key: str, value: str, scope: str = "global"):
    from core.database import save_memory
    save_memory(key, value, source="manual", scope=scope, agent_id="")
    scope_label = "全局" if scope == "global" else "Agent私有"
    console.print(f"[green]✓ 已保存{scope_label}记忆: {key}[/]")


def _memory_delete(identifier: str, scope: str = "all"):
    """
    按序号或键名删除记忆。
    identifier: 数字序号（基于当前列表）或键名
    scope: 'global' / 'agent' / 'all'（序号模式下必须指定具体 scope）
    """
    from core.database import (
        load_memories, load_all_memories,
        delete_memory_by_id, delete_memory_by_key, find_memory_by_identifier
    )

    if identifier.isdigit():
        # 序号模式：先获取对应列表
        if scope == "all":
            memories = load_all_memories(agent_id="")
        else:
            memories = load_memories(scope=scope, agent_id="")
        idx = int(identifier) - 1
        if not (0 <= idx < len(memories)):
            console.print(f"[red]序号 {identifier} 超出范围（共 {len(memories)} 条）[/]")
            return
        target = memories[idx]
        delete_memory_by_id(target["id"])
        console.print(f"[green]✓ 已删除记忆: {target['key']}[/]")
    else:
        # 键名模式
        if scope == "all":
            # 键名模式下尝试全局，找不到再试私有
            from core.database import find_memory_by_identifier as _find
            target = _find(identifier, scope="global", agent_id="")
            if not target:
                target = _find(identifier, scope="agent", agent_id="")
            if not target:
                console.print(f"[red]找不到记忆键: {identifier}[/]")
                return
            delete_memory_by_id(target["id"])
        else:
            delete_memory_by_key(identifier, scope=scope, agent_id="")
        console.print(f"[green]✓ 已删除记忆: {identifier}[/]")


def _handle_memory_cmd(parts: list[str]):
    sub = parts[1] if len(parts) > 1 else "list"

    if sub in ("list", "ls"):
        # /memory list [global|agent|all]
        scope = parts[2] if len(parts) > 2 else "all"
        _memory_list(scope)

    elif sub == "set" and len(parts) >= 4:
        # /memory set <键> <值> [global|agent]
        scope = parts[-1] if parts[-1] in ("global", "agent") else "global"
        if parts[-1] in ("global", "agent"):
            value = " ".join(parts[3:-1])
        else:
            value = " ".join(parts[3:])
        _memory_set(parts[2], value, scope)

    elif sub in ("delete", "rm", "del") and len(parts) >= 3:
        # /memory delete <序号/键名> [global|agent|all]
        scope = parts[3] if len(parts) > 3 and parts[3] in ("global", "agent", "all") else "all"
        _memory_delete(parts[2], scope)

    else:
        console.print(Panel(
            "/memory list [global|agent|all]          列出记忆（默认 all）\n"
            "/memory set <键> <值> [global|agent]     保存记忆（默认 global）\n"
            "/memory delete <序号/键名> [scope]        按序号或键名删除",
            title="memory 指令",
        ))


# ──────────────────────────────────────────
# /task 指令
# ──────────────────────────────────────────

def _task_submit(description: str):
    from core.task_runner import submit_task
    tid = submit_task(description, _current_agent_id)
    console.print(f"[green]✓ 任务已提交，ID: [bold]{tid}[/]（用 /task status {tid} 查看进度）[/]")


def _task_list():
    from core.task_runner import list_all_tasks
    import time
    tasks = list_all_tasks()
    if not tasks:
        console.print("[dim]暂无任务[/]")
        return
    table = Table(title="后台任务", border_style="cyan")
    table.add_column("ID", style="bold", width=10)
    table.add_column("描述", max_width=40)
    table.add_column("状态")
    table.add_column("创建时间")
    STATUS_COLOR = {"pending": "yellow", "running": "cyan", "done": "green", "error": "red"}
    for t in tasks:
        color = STATUS_COLOR.get(t["status"], "white")
        ts = time.strftime("%m-%d %H:%M", time.localtime(t["created_at"]))
        table.add_row(t["id"], t["description"][:40], f"[{color}]{t['status']}[/]", ts)
    console.print(table)


def _task_status(tid: str):
    from core.task_runner import get_task_status
    t = get_task_status(tid)
    if not t:
        console.print(f"[red]找不到任务 {tid}[/]")
        return
    console.print(Panel(
        f"描述: {t['description']}\n"
        f"状态: {t['status']}\n"
        f"结果: {t['result'] or '(进行中)'}\n"
        f"错误: {t['error'] or '无'}",
        title=f"任务 {t['id']}",
    ))


def _handle_task_cmd(parts: list[str]):
    sub = parts[1] if len(parts) > 1 else "list"
    if sub in ("list", "ls"):
        _task_list()
    elif sub in ("run", "submit") and len(parts) > 2:
        _task_submit(" ".join(parts[2:]))
    elif sub in ("status", "show") and len(parts) > 2:
        _task_status(parts[2])
    else:
        console.print(Panel(
            "/task list                      列出所有任务\n"
            "/task run <任务描述>             提交后台任务\n"
            "/task status <ID>               查看任务状态",
            title="task 指令",
        ))


# ──────────────────────────────────────────
# /provider 指令
# ──────────────────────────────────────────

def _provider_list():
    registry = _cfg.providers
    active = registry.active_name()
    table = Table(title="LLM Providers", border_style="cyan")
    table.add_column("", width=2)
    table.add_column("名称", style="bold")
    table.add_column("类型")
    table.add_column("模型")
    table.add_column("Base URL")
    table.add_column("API Key")
    for name in registry.list():
        p = registry.get(name)
        marker = "[green]●[/]" if name == active else " "
        key_display = f"env:{p.api_key_env}" if p.api_key_env else ("[dim]明文[/]" if p.api_key else "[red]未配置[/]")
        table.add_row(marker, name, p.type, p.model, p.base_url or "[dim]-[/]", key_display)
    console.print(table)
    console.print(f"[dim]当前激活: [bold]{active}[/][/]")


def _provider_use(name: str):
    if _cfg.providers.set_active(name):
        _rebuild_agent()
        p = _cfg.providers.active()
        console.print(f"[green]✓ 已切换到 {name} ({p.type} / {p.model})[/]")
    else:
        console.print(f"[red]Provider '{name}' 不存在[/]")


def _provider_add():
    from rich.prompt import Prompt
    console.print("[bold]添加新 Provider[/]\n")
    name = Prompt.ask("名称").strip()
    if not name:
        return
    ptype = Prompt.ask("类型", choices=["anthropic", "openai", "ollama"], default="openai")
    model = Prompt.ask("模型名称", default="")
    console.print("[dim]推荐填写环境变量名（Key 存在容器外宿主机）[/]")
    api_key_env = Prompt.ask("API Key 环境变量名（推荐）", default="")
    api_key = ""
    if not api_key_env:
        api_key = Prompt.ask("或直接填写 API Key（不推荐）", default="", password=True)
    base_url = Prompt.ask("Base URL（官方端点可留空）", default="")
    max_tokens = int(Prompt.ask("Max Tokens", default="8192"))
    from core.config import ProviderProfile
    profile = ProviderProfile(
        name=name, type=ptype, api_key=api_key, api_key_env=api_key_env,
        base_url=base_url, model=model, max_tokens=max_tokens,
    )
    _cfg.providers.add(profile)
    console.print(f"[green]✓ Provider '{name}' 已保存[/]")
    if Confirm.ask(f"立即切换到 {name}？", default=True):
        _provider_use(name)


def _handle_provider_cmd(parts: list[str]):
    sub = parts[1] if len(parts) > 1 else "list"
    if sub in ("list", "ls"):
        _provider_list()
    elif sub in ("use", "switch") and len(parts) > 2:
        _provider_use(parts[2])
    elif sub == "add":
        _provider_add()
    else:
        console.print(Panel(
            "/provider list           列出所有 provider\n"
            "/provider use <名称>     切换 provider\n"
            "/provider add            添加新 provider",
            title="provider 指令",
        ))


# ──────────────────────────────────────────
# 斜杠指令总路由
# ──────────────────────────────────────────

# ──────────────────────────────────────────
# /agent 指令
# ──────────────────────────────────────────

def _agent_list():
    """列出所有 Agent 配置，标记运行中和当前交互的"""
    from core.agent_registry import agent_registry
    profiles = agent_registry.profiles()
    if not profiles:
        console.print("[dim]agents/ 目录下暂无 Agent 配置[/]")
        return
    table = Table(title="Agent 列表", border_style="cyan")
    table.add_column("", width=2)
    table.add_column("ID", style="bold")
    table.add_column("名称")
    table.add_column("Provider", style="dim")
    table.add_column("Skills", style="dim")
    table.add_column("状态")
    for p in profiles:
        is_current = p.id == _current_agent_id
        is_running = agent_registry.is_running(p.id)
        marker = "[green]▶[/]" if is_current else " "
        provider_str = p.provider or "[dim]全局默认[/]"
        skills_str = ", ".join(p.skills) if p.skills else "[dim]-[/]"
        status = "[green]运行中[/]" if is_running else "[dim]未启动[/]"
        table.add_row(marker, p.id, p.name, provider_str, skills_str, status)
    console.print(table)
    console.print(f"[dim]▶ 当前交互: [bold]{_current_agent_id}[/][/]")


def _agent_switch(aid: str) -> str:
    """切换当前交互的 Agent，必要时自动启动"""
    global _current_agent_id, _agent, _llm
    from core.agent_registry import agent_registry

    if not agent_registry.get_profile(aid):
        console.print(f"[red]Agent '{aid}' 配置不存在，请检查 agents/{aid}.yaml[/]")
        return _current_agent_id

    # 启动（如果尚未运行）
    if not agent_registry.is_running(aid):
        with console.status(f"[cyan]启动 Agent [{aid}]...", spinner="dots"):
            try:
                runtime = agent_registry.start(aid, _cfg, _kb, _all_tools)
            except Exception as e:
                console.print(f"[red]Agent [{aid}] 启动失败: {e}[/]")
                return _current_agent_id
    else:
        runtime = agent_registry.get_runtime(aid)

    _current_agent_id = aid
    _agent = runtime
    _llm = runtime.llm
    profile = agent_registry.get_profile(aid)
    console.print(f"[green]✓ 已切换到 Agent: {profile.name} [{aid}][/]")
    return aid


def _agent_start(aid: str):
    """后台启动一个 Agent（不切换当前交互）"""
    from core.agent_registry import agent_registry
    if agent_registry.is_running(aid):
        console.print(f"[yellow]Agent [{aid}] 已在运行中[/]")
        return
    if not agent_registry.get_profile(aid):
        console.print(f"[red]Agent '{aid}' 配置不存在[/]")
        return
    with console.status(f"[cyan]启动 Agent [{aid}]...", spinner="dots"):
        try:
            agent_registry.start(aid, _cfg, _kb, _all_tools)
            profile = agent_registry.get_profile(aid)
            console.print(f"[green]✓ Agent [{aid}] ({profile.name}) 已在后台启动[/]")
        except Exception as e:
            console.print(f"[red]Agent [{aid}] 启动失败: {e}[/]")


def _agent_stop(aid: str):
    """停止一个 Agent"""
    global _current_agent_id
    from core.agent_registry import agent_registry
    if aid == _current_agent_id:
        console.print(f"[red]无法停止当前正在交互的 Agent，请先切换到其他 Agent[/]")
        return
    if agent_registry.stop(aid):
        console.print(f"[green]✓ Agent [{aid}] 已停止[/]")
    else:
        console.print(f"[yellow]Agent [{aid}] 未在运行[/]")


def _agent_show(aid: str):
    """查看 Agent 配置详情"""
    from core.agent_registry import agent_registry
    profile = agent_registry.get_profile(aid)
    if profile is None:
        console.print(f"[red]Agent '{aid}' 不存在[/]")
        return
    is_running = agent_registry.is_running(aid)
    status = "[green]运行中[/]" if is_running else "[dim]未启动[/]"
    tools_str = ", ".join(profile.base_tools) if profile.base_tools else "(全部)"
    skills_str = ", ".join(profile.skills) if profile.skills else "(无)"
    prompt_preview = (profile.system_prompt[:200] + "..."
                      if len(profile.system_prompt) > 200
                      else profile.system_prompt or "(无)")
    console.print(Panel(
        f"ID: [bold]{profile.id}[/]  {status}\n"
        f"名称: {profile.name}\n"
        f"描述: {profile.description or '(无)'}\n"
        f"Provider: {profile.provider or '全局默认'}\n"
        f"工作目录: {profile.workdir or '全局默认'}\n"
        f"基础工具: {tools_str}\n"
        f"Skills: {skills_str}\n"
        f"全局记忆: {'✓' if profile.memory.use_global else '✗'}  "
        f"私有记忆: {'✓' if profile.memory.use_private else '✗'}\n"
        f"System Prompt 预览:\n{prompt_preview}",
        title="Agent 详情",
    ))


def _agent_reload():
    """重新扫描 agents/ 目录（不影响已运行的 Agent）"""
    from core.agent_registry import agent_registry
    agent_registry.reload_profiles()
    count = len(agent_registry.profiles())
    console.print(f"[green]✓ 已重载 Agent 配置（共 {count} 个）[/]")


def _handle_agent_cmd(parts: list[str]) -> str:
    """返回切换后的 agent_id（如果没有切换则返回空字符串）"""
    sub = parts[1] if len(parts) > 1 else "list"
    if sub in ("list", "ls"):
        _agent_list()
    elif sub in ("switch", "use") and len(parts) >= 3:
        return _agent_switch(parts[2])
    elif sub == "start" and len(parts) >= 3:
        _agent_start(parts[2])
    elif sub == "stop" and len(parts) >= 3:
        _agent_stop(parts[2])
    elif sub == "show" and len(parts) >= 3:
        _agent_show(parts[2])
    elif sub == "reload":
        _agent_reload()
    else:
        console.print(Panel(
            "/agent list                  列出所有 Agent（▶ 标记当前）\n"
            "/agent switch <ID>           切换当前交互的 Agent\n"
            "/agent start  <ID>           后台启动 Agent（不切换）\n"
            "/agent stop   <ID>           停止 Agent\n"
            "/agent show   <ID>           查看 Agent 配置详情\n"
            "/agent reload                重新扫描 agents/ 目录",
            title="agent 指令",
        ))
    return ""


# ──────────────────────────────────────────
# /skill 指令
# ──────────────────────────────────────────

def _skill_list():
    """列出所有可用 Skill，标记已激活的"""
    from core.skill_loader import skill_registry
    skills = skill_registry.all()
    if not skills:
        console.print("[dim]skills/ 目录下暂无 Skill（example.yaml 不计入）[/]")
        console.print("[dim]新建 skills/<名称>.yaml 即可添加[/]")
        return
    table = Table(title="可用 Skill", border_style="cyan")
    table.add_column("", width=2)
    table.add_column("ID", style="bold")
    table.add_column("名称")
    table.add_column("描述")
    table.add_column("工具", style="dim")
    table.add_column("知识", style="dim")
    for s in skills:
        marker = "[green]●[/]" if s.id in _active_skills else " "
        tools_str = ", ".join(s.tools) if s.tools else "-"
        know_str = ", ".join(s.knowledge) if s.knowledge else "-"
        table.add_row(marker, s.id, s.name, s.description or "-",
                      tools_str, know_str)
    console.print(table)
    if _active_skills:
        console.print(f"[dim]当前激活: {', '.join(_active_skills)}[/]")


def _skill_enable(sid: str):
    from core.skill_loader import skill_registry
    if not skill_registry.exists(sid):
        console.print(f"[red]Skill '{sid}' 不存在，请检查 skills/{sid}.yaml[/]")
        return
    if sid in _active_skills:
        console.print(f"[yellow]Skill '{sid}' 已处于激活状态[/]")
        return
    _active_skills.append(sid)
    _save_active_skills(_active_skills)
    _rebuild_agent_with_skills()
    skill = skill_registry.get(sid)
    console.print(f"[green]✓ 已激活 Skill: {skill.name}[/]")


def _skill_disable(sid: str):
    if sid not in _active_skills:
        console.print(f"[yellow]Skill '{sid}' 未激活[/]")
        return
    _active_skills.remove(sid)
    _save_active_skills(_active_skills)
    _rebuild_agent_with_skills()
    console.print(f"[green]✓ 已停用 Skill: {sid}[/]")


def _skill_show(sid: str):
    from core.skill_loader import skill_registry
    skill = skill_registry.get(sid)
    if skill is None:
        console.print(f"[red]Skill '{sid}' 不存在[/]")
        return
    status = "[green]已激活[/]" if sid in _active_skills else "[dim]未激活[/]"
    tools_str = "\n  ".join(skill.tools) if skill.tools else "(无)"
    know_str = "\n  ".join(skill.knowledge) if skill.knowledge else "(无)"
    prompt_preview = skill.system_prompt[:200] + "..." if len(skill.system_prompt) > 200 else skill.system_prompt
    console.print(Panel(
        f"ID: [bold]{skill.id}[/]  {status}\n"
        f"名称: {skill.name}\n"
        f"描述: {skill.description or '(无)'}\n"
        f"工具:\n  {tools_str}\n"
        f"知识集合:\n  {know_str}\n"
        f"System Prompt 预览:\n{prompt_preview}",
        title="Skill 详情",
    ))


def _skill_reload():
    from core.skill_loader import skill_registry
    skill_registry.reload()
    _rebuild_agent_with_skills()
    console.print(f"[green]✓ 已重载所有 Skill（共 {len(skill_registry.all())} 个）[/]")


def _handle_skill_cmd(parts: list[str]):
    sub = parts[1] if len(parts) > 1 else "list"
    if sub in ("list", "ls"):
        _skill_list()
    elif sub in ("enable", "on") and len(parts) >= 3:
        _skill_enable(parts[2])
    elif sub in ("disable", "off") and len(parts) >= 3:
        _skill_disable(parts[2])
    elif sub == "show" and len(parts) >= 3:
        _skill_show(parts[2])
    elif sub == "reload":
        _skill_reload()
    else:
        console.print(Panel(
            "/skill list                  列出所有 Skill（● 标记已激活）\n"
            "/skill enable  <ID>          激活 Skill\n"
            "/skill disable <ID>          停用 Skill\n"
            "/skill show    <ID>          查看 Skill 详情\n"
            "/skill reload                重新扫描 skills/ 目录",
            title="skill 指令",
        ))


# ──────────────────────────────────────────
# /cache 指令
# ──────────────────────────────────────────

def _handle_cache_cmd(parts: list[str]):
    """查看或管理工具调用缓存"""
    from core.agent_registry import agent_registry
    sub = parts[1] if len(parts) > 1 else "status"
    runtime = agent_registry.get_runtime(_current_agent_id)

    if sub == "status":
        if runtime is None or runtime._cache is None:
            console.print("[dim]当前 Agent 无缓存实例[/]")
            return
        stats = runtime._cache.stats()
        table = Table(title=f"工具缓存状态（Agent: {_current_agent_id}）",
                      border_style="cyan")
        table.add_column("指标")
        table.add_column("值", style="bold")
        for k, v in stats.items():
            table.add_row(k, str(v))
        console.print(table)

    elif sub == "clear":
        if runtime is None or runtime._cache is None:
            console.print("[dim]当前 Agent 无缓存实例[/]")
            return
        runtime._cache.invalidate()
        console.print("[green]✓ 缓存已清空[/]")

    else:
        console.print(Panel(
            "/cache status    查看缓存命中统计\n"
            "/cache clear     清空当前 Agent 的全部缓存",
            title="cache 指令",
        ))


def _handle_rag_cmd(parts: list[str]):
    """RAG 知识库管理"""
    sub = parts[1] if len(parts) > 1 else "status"

    if sub == "status":
        _rag_status()
    elif sub in ("list", "ls"):
        _rag_list()
    elif sub == "new" and len(parts) >= 3:
        _rag_new(" ".join(parts[2:]))
    elif sub in ("delete", "rm") and len(parts) >= 3:
        _rag_delete(parts[2])
    elif sub == "show" and len(parts) >= 3:
        _rag_show(parts[2])
    else:
        console.print(Panel(
            "/rag status              RAG 连接状态\n"
            "/rag list                列出所有知识集合\n"
            "/rag new <名称>          新建知识集合\n"
            "/rag show <序号/ID>      查看集合详情\n"
            "/rag delete <序号/ID>    删除集合及其文档",
            title="RAG 指令",
        ))


def _rag_status():
    if _kb is None:
        console.print(Panel("[red]未连接[/] — 数据库配置未填写或连接失败", title="RAG 状态"))
        return
    state = _kb.state
    color = {"初始化中": "yellow", "初始化完成": "green", "未连接": "red"}.get(state, "white")
    extra = ""
    if state == "未连接" and _kb._init_error:
        extra = f"\n错误: {_kb._init_error}"
    if state == "初始化完成":
        from core.database import list_collections
        count = len(list_collections())
        extra = f"\n数据库: {_kb._cfg.host}:{_kb._cfg.port}/{_kb._cfg.db}  |  知识集合: {count} 个"
    console.print(Panel(f"[{color}]{state}[/]{extra}", title="RAG 状态"))


def _rag_list():
    from core.database import list_collections
    import time
    cols = list_collections()
    if not cols:
        console.print("[dim]暂无知识集合，使用 /rag new <名称> 创建[/]")
        return
    table = Table(title="知识集合", border_style="cyan")
    table.add_column("#", width=4, style="bold cyan")
    table.add_column("ID", style="dim", width=10)
    table.add_column("名称")
    table.add_column("文档块数", justify="right")
    table.add_column("更新时间")
    for i, c in enumerate(cols, 1):
        ts = time.strftime("%m-%d %H:%M", time.localtime(c["updated_at"]))
        table.add_row(str(i), c["id"], c["name"], str(c["doc_count"]), ts)
    console.print(table)


def _rag_new(name: str):
    from core.database import create_collection
    desc = ""
    cid = create_collection(name, desc)
    console.print(f"[green]✓ 已创建知识集合 '{name}'，ID: [bold]{cid}[/][/]")
    console.print(f"[dim]导入文档：/ingest <路径> {cid}[/]")


def _rag_show(identifier: str):
    from core.database import find_collection_by_identifier
    import time
    c = find_collection_by_identifier(identifier)
    if not c:
        console.print(f"[red]找不到集合 '{identifier}'[/]")
        return
    ts_c = time.strftime("%Y-%m-%d %H:%M", time.localtime(c["created_at"]))
    ts_u = time.strftime("%Y-%m-%d %H:%M", time.localtime(c["updated_at"]))
    console.print(Panel(
        f"ID: [bold]{c['id']}[/]\n"
        f"名称: {c['name']}\n"
        f"描述: {c['description'] or '(无)'}\n"
        f"文档块数: {c['doc_count']}\n"
        f"创建时间: {ts_c}\n"
        f"更新时间: {ts_u}",
        title="知识集合详情",
    ))


def _rag_delete(identifier: str):
    from core.database import find_collection_by_identifier, delete_collection
    c = find_collection_by_identifier(identifier)
    if not c:
        console.print(f"[red]找不到集合 '{identifier}'[/]")
        return
    if not Confirm.ask(f"确认删除集合 '{c['name']}' 及其所有文档？", default=False):
        return
    if _kb:
        _kb.delete_collection_docs(c["id"])
    delete_collection(c["id"])
    console.print(f"[green]✓ 已删除集合 '{c['name']}'[/]")


def _handle_slash(cmd: str, session_id: str, tools: list) -> str:
    """返回（可能更新的）session_id"""
    parts = cmd.strip().split()
    name = parts[0].lower()

    if name in ("/quit", "/exit", "/q"):
        console.print("[dim]再见！[/]")
        sys.exit(0)

    elif name == "/help":
        console.print(Panel(
            "/help                         显示帮助\n"
            "/provider [子命令]             管理 LLM Provider\n"
            "/session  [子命令]             管理对话记录（list/new/load/delete）\n"
            "/memory   [子命令]             管理长期记忆（list/set/delete，支持序号）\n"
            "/task     [子命令]             后台任务（list/run/status）\n"
            "/tools                        列出所有工具\n"
            "/agent [list/switch/start/stop/show] 管理 Agent\n"
            "/skill [list/enable/disable/show]   管理 Skill\n"
            "/cache [status/clear]              工具调用缓存\n"
            "/rag [status/list/new/show/delete]  管理 RAG 知识集合\n"
            "/ingest <路径> [集合ID]        导入文档到知识库\n"
            "/clear                        清空当前对话记忆（新建会话）\n"
            "/quit                         退出",
            title="指令列表",
        ))

    elif name == "/provider":
        _handle_provider_cmd(parts)

    elif name == "/session":
        return _handle_session_cmd(parts, session_id)

    elif name == "/memory":
        _handle_memory_cmd(parts)

    elif name == "/task":
        _handle_task_cmd(parts)

    elif name == "/tools":
        table = Table(title=f"可用工具 ({len(tools)})", border_style="cyan")
        table.add_column("工具名", style="bold cyan")
        table.add_column("来源", style="dim")
        table.add_column("描述")
        for t in tools:
            source = getattr(t, "_source", "-")
            desc = (t.description or "").split("\n")[0][:60]
            table.add_row(t.name, source, desc)
        console.print(table)

    elif name == "/clear":
        new_sid = _session_new()
        console.print("[yellow]已开启新对话[/]")
        return new_sid

    elif name == "/agent":
        old_aid = _current_agent_id
        new_aid = _handle_agent_cmd(parts)
        # 如果 Agent 切换了，返回 None 信号让 chat 重置 session_id
        if _current_agent_id != old_aid:
            return None

    elif name == "/skill":
        _handle_skill_cmd(parts)

    elif name == "/cache":
        _handle_cache_cmd(parts)

    elif name == "/rag":
        _handle_rag_cmd(parts)

    elif name == "/ingest":
        if len(parts) < 2:
            console.print("[red]用法: /ingest <路径> [集合ID]\n先用 /rag new <名称> 创建集合[/]")
        elif _kb is None:
            console.print("[red]RAG 未启用[/]")
        else:
            collection_id = parts[2] if len(parts) >= 3 else ""
            if collection_id:
                from core.database import get_collection
                col = get_collection(collection_id)
                if not col:
                    console.print(f"[red]找不到集合 ID '{collection_id}'，请先用 /rag new 创建[/]")
                    return session_id
                console.print(f"[dim]导入到集合: {col['name']} ({collection_id})[/]")
            with console.status(f"导入 {parts[1]}..."):
                n = _kb.ingest(parts[1], collection_id=collection_id)
            console.print(f"[green]✓ 导入完成：{n} 个文本块[/]")

    else:
        console.print(f"[yellow]未知指令: {name}，输入 /help 查看[/]")

    return session_id


# ──────────────────────────────────────────
# CLI 命令
# ──────────────────────────────────────────

@click.group(invoke_without_command=True)
@click.argument("query", required=False)
@click.pass_context
def cli(ctx, query):
    """🤖 本地 LangGraph Agent"""
    if ctx.invoked_subcommand is None:
        if query:
            _bootstrap()
            from core.database import create_session
            sid = create_session()
            with console.status("[bold cyan]思考中...", spinner="dots"):
                answer = _run_query(query, sid)
            console.print(Markdown(answer))
        else:
            ctx.invoke(chat)


@cli.command()
def chat():
    """交互式对话（默认模式）"""
    tools = _bootstrap()
    session_id = None  # 延迟创建：首次发消息时才建 session，避免空记录
    console.print(f"[dim]当前 Agent: [bold]{_current_agent_id}[/]。↑↓ 翻历史，Tab 补全，/help 查看指令。[/]")

    # 输入历史（↑↓ 翻历史）
    _history = InMemoryHistory()

    # 斜杠指令补全
    _slash_commands = [
        "/help", "/quit", "/exit",
        "/session", "/session list", "/session new", "/session load", "/session delete",
        "/memory", "/memory list", "/memory list global", "/memory list agent",
        "/memory set", "/memory delete",
        "/task", "/task list", "/task run", "/task status",
        "/provider", "/provider list", "/provider use", "/provider add",
        "/tools", "/ingest", "/clear",
        "/agent", "/agent list", "/agent switch", "/agent start", "/agent stop",
        "/agent show", "/agent reload",
        "/skill", "/skill list", "/skill enable", "/skill disable", "/skill show", "/skill reload",
        "/cache", "/cache status", "/cache clear",
        "/rag", "/rag status", "/rag list", "/rag new", "/rag show", "/rag delete",
    ]
    _completer = WordCompleter(_slash_commands, match_middle=False, sentence=True)

    while True:
        try:
            user_input = pt_prompt(
                "\n你> ",
                history=_history,
                auto_suggest=AutoSuggestFromHistory(),
                completer=_completer,
                complete_while_typing=False,  # 只在按 Tab 时补全，不自动弹出
            ).strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]再见！[/]")
            break
        except UnicodeDecodeError:
            console.print("[yellow]输入编码错误，请重试[/]")
            continue

        if not user_input:
            continue

        if user_input.startswith("/"):
            new_sid = _handle_slash(user_input, session_id or "", tools)
            if new_sid is None:
                # Agent 切换信号：重置 session_id
                session_id = None
                console.print(f"[dim]已切换到 Agent [{_current_agent_id}]，开启新对话[/]")
            elif new_sid and new_sid != (session_id or ""):
                session_id = new_sid
                console.print(f"[dim]当前对话: [bold]{session_id[:8]}[/][/]")
            continue

        # 首次发消息时创建 session（绑定当前 Agent）
        if session_id is None:
            from core.database import create_session
            session_id = create_session(agent_id=_current_agent_id)

        with console.status("[bold cyan]思考中...", spinner="dots"):
            try:
                answer = _run_query(user_input, session_id)
            except Exception as e:
                import traceback
                answer = f"执行出错: {e}\n{traceback.format_exc()}"

        console.print("\n[bold green]Agent[/]")
        console.print(Markdown(answer))


@cli.command()
@click.argument("path")
def ingest(path: str):
    """导入文件或目录到 RAG 知识库"""
    _bootstrap()
    if _kb is None:
        console.print("[red]RAG 未启用[/]")
        sys.exit(1)
    with console.status(f"导入 {path}..."):
        n = _kb.ingest(path)
    console.print(f"[green]✓ {n} 个文本块[/]")


@cli.command()
def tools():
    """列出所有可用工具"""
    all_tools = _bootstrap()
    from rich.table import Table as RichTable
    table = RichTable(title="可用工具", border_style="cyan")
    table.add_column("工具名", style="bold cyan")
    table.add_column("来源", style="dim")
    table.add_column("描述")
    for t in all_tools:
        source = getattr(t, "_source", "-")
        desc = (t.description or "").split("\n")[0][:60]
        table.add_row(t.name, source, desc)
    console.print(table)


@cli.command("provider")
@click.argument("subcommand", default="list")
@click.argument("name", required=False)
def provider_cmd(subcommand, name):
    """管理 LLM Provider"""
    _bootstrap()
    parts = ["/provider", subcommand] + ([name] if name else [])
    _handle_provider_cmd(parts)


if __name__ == "__main__":
    cli()
