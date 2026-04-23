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

console = Console()

_agent = None
_llm = None
_kb = None
_cfg = None
_all_tools = []
_current_session_id = None
_restored_sessions: set = set()  # 已从数据库恢复上下文的 session
_event_loop = None  # 持久化事件循环，保证 InMemorySaver 状态跨调用存活


def _get_loop():
    global _event_loop
    if _event_loop is None or _event_loop.is_closed():
        import asyncio
        _event_loop = asyncio.new_event_loop()
    return _event_loop


def _bootstrap():
    global _agent, _llm, _kb, _cfg, _all_tools

    sys.path.insert(0, str(Path(__file__).parent))
    from core.config import config
    _cfg = config

    log_path = Path(config.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stderr),
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
        from tools.builtin_tools import BUILTIN_TOOLS, set_workspace
        set_workspace(config.workspace_dir)
        all_tools = list(BUILTIN_TOOLS)

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

    profile = config.providers.active()
    console.print(Panel(
        f"[bold green]Agent 就绪[/]\n"
        f"Provider : [cyan]{config.providers.active_name()}[/] "
        f"([dim]{profile.type} / {profile.model}[/])\n"
        f"工作目录 : [cyan]{config.workspace_dir}[/]\n"
        f"RAG 知识库: {'[green]已连接[/]' if kb else '[yellow]未启用[/]'}\n"
        f"工具数量 : [cyan]{len(all_tools)}[/]",
        title="🤖 CLI Agent",
        border_style="cyan",
    ))
    return all_tools


def _rebuild_agent():
    global _agent, _llm
    from core.agent_graph import build_agent
    _agent, _llm = build_agent(_cfg, _kb, _all_tools)


def _run_query(query: str, thread_id: str) -> str:
    from langchain_core.messages import HumanMessage, AIMessage
    from core.database import save_message, load_messages

    # 保存用户消息
    save_message(thread_id, "human", query)

    async def _ainvoke():
        # 如果是已加载的旧对话且尚未恢复上下文，先把历史消息注入
        if thread_id not in _restored_sessions:
            history = load_messages(thread_id)
            # 取除最后一条（刚存入的当前问题）之外的所有历史
            prior = history[:-1]
            if prior:
                prior_msgs = []
                for m in prior:
                    if m["role"] in ("human", "user"):
                        prior_msgs.append(HumanMessage(content=m["content"]))
                    elif m["role"] in ("assistant", "ai"):
                        prior_msgs.append(AIMessage(content=m["content"]))
                if prior_msgs:
                    # 先把历史消息灌入 LangGraph，建立上下文
                    await _agent.ainvoke(
                        {"messages": prior_msgs, "workspace": _cfg.workspace_dir},
                        config={"configurable": {"thread_id": thread_id}},
                    )
            _restored_sessions.add(thread_id)

        result = await _agent.ainvoke(
            {"messages": [HumanMessage(content=query)], "workspace": _cfg.workspace_dir},
            config={"configurable": {"thread_id": thread_id}},
        )
        last = result["messages"][-1]
        return last.content if hasattr(last, "content") else str(last)

    answer = _get_loop().run_until_complete(_ainvoke())

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
    sessions = list_sessions()
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
    sid = create_session(name)
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
    sessions = list_sessions(100)
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
    sessions = list_sessions(100)
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

def _memory_list():
    from core.database import load_all_memories
    memories = load_all_memories()
    if not memories:
        console.print("[dim]暂无长期记忆[/]")
        return
    table = Table(title="长期记忆", border_style="cyan")
    table.add_column("键", style="bold")
    table.add_column("内容")
    table.add_column("来源", style="dim")
    for m in memories:
        table.add_row(m["key"], m["value"], m["source"] or "-")
    console.print(table)


def _memory_set(key: str, value: str):
    from core.database import save_memory
    save_memory(key, value, source="manual")
    console.print(f"[green]✓ 已保存记忆: {key}[/]")


def _memory_delete(key: str):
    from core.database import delete_memory
    delete_memory(key)
    console.print(f"[green]✓ 已删除记忆: {key}[/]")


def _handle_memory_cmd(parts: list[str]):
    sub = parts[1] if len(parts) > 1 else "list"
    if sub in ("list", "ls"):
        _memory_list()
    elif sub == "set" and len(parts) > 3:
        _memory_set(parts[2], " ".join(parts[3:]))
    elif sub in ("delete", "rm") and len(parts) > 2:
        _memory_delete(parts[2])
    else:
        console.print(Panel(
            "/memory list                列出所有记忆\n"
            "/memory set <键> <值>       手动添加记忆\n"
            "/memory delete <键>         删除记忆",
            title="memory 指令",
        ))


# ──────────────────────────────────────────
# /task 指令
# ──────────────────────────────────────────

def _task_submit(description: str):
    from core.task_runner import submit_task
    tid = submit_task(description, _agent, _cfg.workspace_dir)
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
            "/memory   [子命令]             管理长期记忆（list/set/delete）\n"
            "/task     [子命令]             后台任务（list/run/status）\n"
            "/tools                        列出所有工具\n"
            "/ingest <路径>                导入文档到知识库\n"
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
        lines = [f"  [cyan]{t.name}[/]: {(t.description or '')[:80]}" for t in tools]
        console.print(Panel("\n".join(lines), title=f"可用工具 ({len(tools)})"))

    elif name == "/clear":
        new_sid = _session_new()
        console.print("[yellow]已开启新对话[/]")
        return new_sid

    elif name == "/ingest":
        if len(parts) < 2:
            console.print("[red]用法: /ingest <路径>[/]")
        elif _kb is None:
            console.print("[red]RAG 未启用[/]")
        else:
            with console.status(f"导入 {parts[1]}..."):
                n = _kb.ingest(parts[1])
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
    console.print("[dim]输入问题后回车。[bold]/help[/] 查看指令，[bold]/quit[/] 退出。[/]")

    while True:
        try:
            user_input = pt_prompt("\n你> ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]再见！[/]")
            break
        except UnicodeDecodeError:
            console.print("[yellow]输入编码错误，请重试[/]")
            continue

        if not user_input:
            continue

        if user_input.startswith("/"):
            # /session 指令需要 session_id，若还未创建则传空字符串
            new_sid = _handle_slash(user_input, session_id or "", tools)
            # 如果指令切换/新建了 session，更新 session_id
            if new_sid and new_sid != (session_id or ""):
                session_id = new_sid
                console.print(f"[dim]当前对话: [bold]{session_id[:8]}[/][/]")
            continue

        # 首次发消息时创建 session
        if session_id is None:
            from core.database import create_session
            session_id = create_session()

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
    for t in all_tools:
        console.print(f"[bold cyan]{t.name}[/]: {t.description or '（无描述）'}")


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
