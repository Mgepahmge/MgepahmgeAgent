"""
builtin_tools.py - Agent 内置工具
- 文件系统（读/写/列目录/查找）
- Web 搜索（DuckDuckGo，无需 API Key）
- 网页抓取
- Shell 命令执行（带安全白名单）
"""
from __future__ import annotations
import os
import subprocess
import logging
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# 工作目录限制（从配置注入，见 agent.py）
_WORKSPACE: str = os.path.expanduser("~/workspace")

def set_workspace(path: str):
    global _WORKSPACE
    _WORKSPACE = os.path.expanduser(path)
    Path(_WORKSPACE).mkdir(parents=True, exist_ok=True)

def _safe_path(path: str) -> Path:
    """确保路径在 workspace 内"""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path(_WORKSPACE) / p
    p = p.resolve()
    workspace = Path(_WORKSPACE).resolve()
    if not str(p).startswith(str(workspace)):
        raise PermissionError(f"路径越界：{p} 不在 {workspace} 内")
    return p


# ──────────────────────────────────────────
# 文件系统工具
# ──────────────────────────────────────────

@tool
def read_file(path: str) -> str:
    """读取文件内容。path 可以是相对于 workspace 的路径或绝对路径。"""
    try:
        p = _safe_path(path)
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"ERROR: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """将内容写入文件（自动创建父目录）。"""
    try:
        p = _safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"已写入 {p}（{len(content)} 字符）"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def list_directory(path: str = ".") -> str:
    """列出目录内容（文件名、类型、大小）。"""
    try:
        p = _safe_path(path)
        if not p.is_dir():
            return f"ERROR: {p} 不是目录"
        lines = []
        for item in sorted(p.iterdir()):
            size = f"{item.stat().st_size:>10,} B" if item.is_file() else "         DIR"
            lines.append(f"{'D' if item.is_dir() else 'F'}  {size}  {item.name}")
        return "\n".join(lines) if lines else "（空目录）"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def find_files(pattern: str, directory: str = ".") -> str:
    """在目录中按 glob 模式查找文件，例如 pattern='**/*.py'。"""
    try:
        p = _safe_path(directory)
        matches = list(p.glob(pattern))
        if not matches:
            return "未找到匹配文件"
        return "\n".join(str(m.relative_to(Path(_WORKSPACE))) for m in matches[:50])
    except Exception as e:
        return f"ERROR: {e}"


# ──────────────────────────────────────────
# Web 工具
# ──────────────────────────────────────────

@tool
def web_search(query: str, max_results: int = 8) -> str:
    """搜索互联网，返回标题 + 摘要 + URL。优先 Brave Search，回退 DuckDuckGo lite。"""
    import os

    # ── 优先：Brave Search（国内可访问，效果好）
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

    # ── 回退：DuckDuckGo lite（不走 Bing，国内可用）
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
        # 截断避免 token 爆炸
        return text[:12000] + ("\n... [截断]" if len(text) > 12000 else "")
    except Exception as e:
        return f"ERROR: {e}"


# ──────────────────────────────────────────
# Shell 工具
# ──────────────────────────────────────────

# 允许执行的命令前缀白名单
_ALLOWED_CMDS = {
    "python", "python3", "pip", "pip3",
    "git", "grep", "find", "cat", "ls", "pwd",
    "head", "tail", "wc", "diff", "sort", "uniq",
    "echo", "mkdir", "cp", "mv", "rm",
    "nvidia-smi", "nvcc",               # CUDA 工具
    "psql",                              # 数据库客户端
    "curl", "wget",
    "make", "cmake",
    "gcc", "g++",
}

@tool
def run_shell(command: str, workdir: Optional[str] = None, timeout: int = 60) -> str:
    """
    在 workspace 中执行 shell 命令。
    出于安全考虑，仅允许白名单内的命令前缀。
    timeout 单位为秒，默认 60。
    """
    # 安全检查
    first_token = command.strip().split()[0].split("/")[-1]
    if first_token not in _ALLOWED_CMDS:
        return (
            f"ERROR: 命令 '{first_token}' 不在允许列表中。\n"
            f"允许的命令: {', '.join(sorted(_ALLOWED_CMDS))}"
        )

    cwd = str(_safe_path(workdir)) if workdir else _WORKSPACE
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


# 导出所有内置工具
BUILTIN_TOOLS = [
    read_file,
    write_file,
    list_directory,
    find_files,
    web_search,
    fetch_url,
    run_shell,
]
