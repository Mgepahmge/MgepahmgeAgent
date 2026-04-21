"""
builtin_tools.py - Agent 内置工具
- 文件系统：全局访问，无路径限制
- Web 搜索：Brave Search 优先，DuckDuckGo lite 回退
- 网页抓取
- Shell：完整 shell 访问
"""
from __future__ import annotations
import os
import subprocess
import logging
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# workspace 目录：Agent 与宿主机交换文件的接口
_WORKSPACE: str = "/workspace"

def set_workspace(path: str):
    global _WORKSPACE
    _WORKSPACE = os.path.expanduser(path)
    Path(_WORKSPACE).mkdir(parents=True, exist_ok=True)


def _resolve_path(path: str) -> Path:
    """
    解析路径：
    - 绝对路径直接使用
    - 相对路径相对于 workspace
    """
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path(_WORKSPACE) / p
    return p.resolve()


# ──────────────────────────────────────────
# 文件系统工具
# ──────────────────────────────────────────

@tool
def read_file(path: str) -> str:
    """读取文件内容。支持绝对路径或相对于 workspace 的路径。"""
    try:
        p = _resolve_path(path)
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"ERROR: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """将内容写入文件（自动创建父目录）。覆盖已有文件前请先向用户说明。"""
    try:
        p = _resolve_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"已写入 {p}（{len(content)} 字符）"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def list_directory(path: str = ".") -> str:
    """列出目录内容（文件名、类型、大小）。支持系统任意目录。"""
    try:
        p = _resolve_path(path)
        if not p.is_dir():
            return f"ERROR: {p} 不是目录"
        lines = []
        for item in sorted(p.iterdir()):
            try:
                size = f"{item.stat().st_size:>10,} B" if item.is_file() else "         DIR"
            except PermissionError:
                size = "  [无权限]"
            lines.append(f"{'D' if item.is_dir() else 'F'}  {size}  {item.name}")
        return "\n".join(lines) if lines else "（空目录）"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def find_files(pattern: str, directory: str = ".") -> str:
    """在目录中按 glob 模式递归查找文件，例如 pattern='**/*.py'。"""
    try:
        p = _resolve_path(directory)
        matches = list(p.glob(pattern))
        if not matches:
            return "未找到匹配文件"
        return "\n".join(str(m) for m in matches[:100])
    except Exception as e:
        return f"ERROR: {e}"


@tool
def delete_file(path: str) -> str:
    """
    删除文件或空目录。
    【警告】此操作不可逆，调用前必须已获得用户明确确认。
    """
    try:
        p = _resolve_path(path)
        if p.is_dir():
            p.rmdir()
            return f"已删除目录 {p}"
        else:
            p.unlink()
            return f"已删除文件 {p}"
    except Exception as e:
        return f"ERROR: {e}"


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
def run_shell(command: str, workdir: Optional[str] = None, timeout: int = 120) -> str:
    """
    执行 shell 命令。可访问整个系统。
    删除、覆盖、卸载等破坏性命令执行前必须已获得用户明确确认。
    timeout 单位为秒，默认 120，长任务可指定更大的值。
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
        if len(output) > 10000:
            output = output[:10000] + "\n... [输出截断]"
        return output or "（命令执行完毕，无输出）"
    except subprocess.TimeoutExpired:
        return f"ERROR: 命令超时（>{timeout}s）"
    except Exception as e:
        return f"ERROR: {e}"


# 导出所有内置工具
BUILTIN_TOOLS = [
    read_file,
    write_file,
    list_directory,
    find_files,
    delete_file,
    web_search,
    fetch_url,
    run_shell,
]
