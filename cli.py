"""
cli.py - 命令行入口，支持多 Provider 管理
"""
from __future__ import annotations
import sys
import uuid
import logging
import click
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import print as rprint

console = Console()

_agent = None
_kb = None
_cfg = None
_all_tools = []


def _bootstrap():
    global _agent, _kb, _cfg, _all_tools

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
        kb = None
        if config.rag.enabled:
            from rag.knowledge_base import KnowledgeBase
            kb = KnowledgeBase.connect(config.rag)
        _kb = kb

        from tools.builtin_tools import BUILTIN_TOOLS, set_workspace
        set_workspace(config.workspace_dir)
        all_tools = list(BUILTIN_TOOLS)

        if config.mcp.servers:
            from tools.mcp_loader import load_mcp_tools_sync
            all_tools.extend(load_mcp_tools_sync(config.mcp))

        _all_tools = all_tools

        from core.agent_graph import build_agent
        _agent = build_agent(config, kb, all_tools)

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
    """切换 provider 后重新构建 agent（不重新加载工具）"""
    global _agent
    from core.agent_graph import build_agent
    _agent = build_agent(_cfg, _kb, _all_tools)


def _run_query(query: str, thread_id: str) -> str:
    from langchain_core.messages import HumanMessage
    result = _agent.invoke(
        {"messages": [HumanMessage(content=query)], "workspace": _cfg.workspace_dir},
        config={"configurable": {"thread_id": thread_id}},
    )
    last = result["messages"][-1]
    return last.content if hasattr(last, "content") else str(last)


# ──────────────────────────────────────────
# /provider 指令处理
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
        key_display = (p.api_key[:8] + "…") if p.api_key else "[dim]无[/]"
        table.add_row(marker, name, p.type, p.model, p.base_url or "[dim]-[/]", key_display)

    console.print(table)
    console.print(f"[dim]当前激活: [bold]{active}[/][/]")


def _provider_use(name: str):
    if _cfg.providers.set_active(name):
        _rebuild_agent()
        p = _cfg.providers.active()
        console.print(f"[green]✓ 已切换到 {name} ({p.type} / {p.model})[/]")
    else:
        console.print(f"[red]Provider '{name}' 不存在，用 /provider list 查看[/]")


def _provider_add():
    """交互式添加 provider"""
    console.print("[bold]添加新 Provider[/]（直接回车使用括号内默认值）\n")

    name = Prompt.ask("名称（唯一标识，如 my-qwen）").strip()
    if not name:
        console.print("[red]名称不能为空[/]")
        return

    ptype = Prompt.ask("类型", choices=["anthropic", "openai", "ollama"], default="openai")
    model = Prompt.ask("模型名称", default="")
    api_key = Prompt.ask("API Key（本地模型可留空）", default="", password=True)
    base_url = Prompt.ask("Base URL（使用官方端点可留空）", default="")
    max_tokens = int(Prompt.ask("Max Tokens", default="8192"))

    from core.config import ProviderProfile
    profile = ProviderProfile(
        name=name, type=ptype, api_key=api_key,
        base_url=base_url, model=model, max_tokens=max_tokens,
    )
    _cfg.providers.add(profile)
    console.print(f"[green]✓ Provider '{name}' 已保存[/]")

    if Confirm.ask(f"立即切换到 {name}？", default=True):
        _provider_use(name)


def _provider_remove(name: str):
    if Confirm.ask(f"确认删除 provider '{name}'？", default=False):
        if _cfg.providers.remove(name):
            console.print(f"[green]✓ 已删除 {name}[/]")
            # 如果删的是当前激活的，重建 agent
            _rebuild_agent()
        else:
            console.print(f"[red]Provider '{name}' 不存在[/]")


def _provider_edit(name: str):
    """编辑已有 provider 的字段"""
    p = _cfg.providers.get(name)
    if p is None:
        console.print(f"[red]Provider '{name}' 不存在[/]")
        return
    console.print(f"[bold]编辑 {name}[/]（直接回车保留当前值）\n")
    p.model     = Prompt.ask("模型名称",    default=p.model)
    new_key     = Prompt.ask("API Key",     default="", password=True)
    if new_key:
        p.api_key = new_key
    p.base_url  = Prompt.ask("Base URL",   default=p.base_url)
    p.max_tokens = int(Prompt.ask("Max Tokens", default=str(p.max_tokens)))
    _cfg.providers.add(p)
    console.print(f"[green]✓ '{name}' 已更新[/]")
    if _cfg.providers.active_name() == name:
        _rebuild_agent()


def _handle_provider_cmd(parts: list[str]):
    sub = parts[1] if len(parts) > 1 else "list"

    if sub in ("list", "ls"):
        _provider_list()
    elif sub in ("use", "switch") and len(parts) > 2:
        _provider_use(parts[2])
    elif sub == "add":
        _provider_add()
    elif sub == "remove" and len(parts) > 2:
        _provider_remove(parts[2])
    elif sub == "edit" and len(parts) > 2:
        _provider_edit(parts[2])
    else:
        console.print(Panel(
            "/provider list              列出所有 provider\n"
            "/provider use  <名称>       切换到指定 provider\n"
            "/provider add               交互式添加新 provider\n"
            "/provider edit <名称>       修改已有 provider\n"
            "/provider remove <名称>     删除 provider",
            title="provider 指令",
        ))


# ──────────────────────────────────────────
# 斜杠指令总路由
# ──────────────────────────────────────────

def _handle_slash(cmd: str, thread_id: str, tools: list):
    parts = cmd.strip().split()
    name = parts[0].lower()

    if name in ("/quit", "/exit", "/q"):
        console.print("[dim]再见！[/]")
        sys.exit(0)

    elif name == "/help":
        console.print(Panel(
            "/help                    显示帮助\n"
            "/provider [子命令]       管理 LLM Provider（list/use/add/edit/remove）\n"
            "/tools                   列出所有工具\n"
            "/ingest <路径>           导入文档到知识库\n"
            "/clear                   清空对话记忆\n"
            "/quit                    退出",
            title="指令列表",
        ))

    elif name == "/provider":
        _handle_provider_cmd(parts)

    elif name == "/tools":
        lines = [f"  [cyan]{t.name}[/]: {(t.description or '')[:80]}" for t in tools]
        console.print(Panel("\n".join(lines), title=f"可用工具 ({len(tools)})"))

    elif name == "/clear":
        console.print("[yellow]对话记忆已清空（下一条消息开启新会话）[/]")

    elif name == "/ingest":
        if len(parts) < 2:
            console.print("[red]用法: /ingest <文件或目录路径>[/]")
            return
        if _kb is None:
            console.print("[red]RAG 未启用（检查 .env 数据库配置）[/]")
            return
        with console.status(f"导入 {parts[1]}..."):
            n = _kb.ingest(parts[1])
        console.print(f"[green]✓ 导入完成：{n} 个文本块[/]")

    else:
        console.print(f"[yellow]未知指令: {name}，输入 /help 查看[/]")


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
            thread_id = str(uuid.uuid4())
            with console.status("[bold cyan]思考中...", spinner="dots"):
                answer = _run_query(query, thread_id)
            console.print(Markdown(answer))
        else:
            ctx.invoke(chat)


@cli.command()
def chat():
    """交互式对话（默认模式）"""
    tools = _bootstrap()
    thread_id = str(uuid.uuid4())
    console.print("[dim]输入问题后回车。[bold]/help[/] 查看指令，[bold]/quit[/] 退出。[/]")

    while True:
        try:
            user_input = Prompt.ask("\n[bold blue]你[/]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]再见！[/]")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            _handle_slash(user_input, thread_id, tools)
            continue

        # /clear 后换新 thread
        if user_input == "/clear":
            thread_id = str(uuid.uuid4())
            continue

        with console.status("[bold cyan]思考中...", spinner="dots"):
            try:
                answer = _run_query(user_input, thread_id)
            except Exception as e:
                answer = f"[red]执行出错: {e}[/]"

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
    """管理 LLM Provider（list/use/add/edit/remove）"""
    _bootstrap()
    parts = ["/provider", subcommand] + ([name] if name else [])
    _handle_provider_cmd(parts)


if __name__ == "__main__":
    cli()
