"""
builtin_tools.py - Agent 内置工具
文件系统操作由 MCP filesystem server 负责（避免工具名冲突），此处只保留：
- Web 搜索：Brave Search 优先，DuckDuckGo lite 回退
- 网页抓取
- Shell 命令执行
"""
from __future__ import annotations
import os
import subprocess
import logging
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_WORKSPACE: str = "/workspace"

def set_workspace(path: str):
    global _WORKSPACE
    _WORKSPACE = os.path.expanduser(path)
    Path(_WORKSPACE).mkdir(parents=True, exist_ok=True)

def _resolve_path(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path(_WORKSPACE) / p
    return p.resolve()


# ──────────────────────────────────────────
# Web 工具
# ──────────────────────────────────────────

@tool
def web_search(query: str, max_results: int = 8) -> str:
    """搜索互联网，返回标题 + 摘要 + URL。优先 Brave Search，回退 DuckDuckGo lite。"""
    brave_key = os.getenv("BRAVE_API_KEY", "")
    if brave_key:
        try:
            import requests as _req
            resp = _req.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={"Accept": "application/json", "X-Subscription-Token": brave_key},
                params={"q": query, "count": max_results},
                timeout=15,
            )
            resp.raise_for_status()
            items = resp.json().get("web", {}).get("results", [])
            if items:
                results = [
                    f"### {r.get('title', '')}\n{r.get('description', '')}\nURL: {r.get('url', '')}"
                    for r in items
                ]
                return "\n---\n".join(results)
        except Exception as e:
            logger.warning(f"Brave Search 失败，回退到 DuckDuckGo: {e}")

    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    last_err = None
    for backend in ["lite", "auto"]:
        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results, backend=backend):
                    results.append(
                        f"### {r.get('title', '')}\n"
                        f"{r.get('body', '')}\n"
                        f"URL: {r.get('href', '')}\n"
                    )
            if results:
                return "\n---\n".join(results)
        except Exception as e:
            last_err = e
            logger.warning(f"DuckDuckGo backend={backend} 失败: {e}")

    return f"搜索失败（所有后端均不可用）: {last_err}"


@tool
def fetch_url(url: str, timeout: int = 15) -> str:
    """抓取指定 URL 的网页正文（自动去除 HTML 标签）。"""
    try:
        import requests
        from bs4 import BeautifulSoup
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AgentBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return text[:12000] + ("\n... [截断]" if len(text) > 12000 else "")
    except Exception as e:
        return f"ERROR: {e}"


# ──────────────────────────────────────────
# Shell 工具
# ──────────────────────────────────────────

@tool
def run_shell(command: str, workdir: Optional[str] = None, timeout: int = 60) -> str:
    """
    在系统中执行 shell 命令。可访问整个系统。
    删除、覆盖、卸载等破坏性命令执行前必须已获得用户明确确认。
    timeout 单位为秒，默认 60。
    """
    cwd = str(_resolve_path(workdir)) if workdir else _WORKSPACE
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout + result.stderr
        if len(output) > 8000:
            output = output[:8000] + "\n... [输出截断]"
        return output or "（命令执行完毕，无输出）"
    except subprocess.TimeoutExpired:
        return f"ERROR: 命令超时（>{timeout}s）"
    except Exception as e:
        return f"ERROR: {e}"


# 导出内置工具（文件系统由 MCP filesystem server 负责）
BUILTIN_TOOLS = [
    web_search,
    fetch_url,
    run_shell,
]
