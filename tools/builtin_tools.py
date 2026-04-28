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
    run_shell,
]
