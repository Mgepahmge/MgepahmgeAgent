"""
plugin_loader.py - 工具插件自动发现和加载
tools/ 目录下所有包含 TOOLS = [...] 列表的 .py 文件自动注册为工具插件。
"""
from __future__ import annotations
import importlib.util
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_TOOLS_DIR = Path(__file__).parent
_EXCLUDE = {"__init__", "builtin_tools", "mcp_loader", "plugin_loader"}


def discover_plugins() -> list:
    """
    扫描 tools/ 目录，加载所有插件文件里的 TOOLS 列表。
    插件文件约定：在模块级别定义 TOOLS = [tool1, tool2, ...]
    """
    all_tools = []
    for path in sorted(_TOOLS_DIR.glob("*.py")):
        name = path.stem
        if name in _EXCLUDE or name.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"tools.{name}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            tools = getattr(mod, "TOOLS", None)
            if tools and isinstance(tools, list):
                all_tools.extend(tools)
                logger.info(f"插件 [{name}]: 加载了 {len(tools)} 个工具")
        except Exception as e:
            logger.warning(f"插件 [{name}] 加载失败: {e}")
    return all_tools
