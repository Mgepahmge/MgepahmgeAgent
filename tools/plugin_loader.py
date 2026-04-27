"""
plugin_loader.py - 工具插件自动发现和加载

约定：
  - tools/ 目录下所有包含 TOOLS = [...] 列表的 .py 文件自动注册
  - 每个工具自动打上 _source 属性，值为插件文件名（不含 .py）
  - 例：memory_tools.py 中的工具，_source = "memory_tools"

_source 用途：
  - /tools 命令显示工具来源
  - Skill / Agent 配置中可直接用文件名批量引用该插件的所有工具
"""
from __future__ import annotations
import importlib.util
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_TOOLS_DIR = Path(__file__).parent
_EXCLUDE = {"__init__", "builtin_tools", "mcp_loader", "plugin_loader"}


def _tag_source(tools: list, source: str) -> list:
    """给工具列表中每个工具附加 _source 属性"""
    for t in tools:
        try:
            t._source = source
        except AttributeError:
            pass  # 极少数 frozen 对象，跳过
    return tools


def load_builtin_tools(workspace: str) -> list:
    """
    加载内置工具（builtin_tools.py），来源标记为 'builtin'。
    workspace 路径在此处注入。
    """
    from tools.builtin_tools import BUILTIN_TOOLS, set_workspace
    set_workspace(workspace)
    return _tag_source(list(BUILTIN_TOOLS), "builtin")


def discover_plugins() -> list:
    """
    扫描 tools/ 目录，加载所有插件文件里的 TOOLS 列表。
    每个插件的工具 _source = 插件文件名（不含 .py）。
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
                _tag_source(tools, name)
                all_tools.extend(tools)
                logger.info(f"插件 [{name}]: 加载了 {len(tools)} 个工具")
        except Exception as e:
            logger.warning(f"插件 [{name}] 加载失败: {e}")
    return all_tools
