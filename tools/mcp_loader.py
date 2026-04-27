"""
mcp_loader.py - 使用 MultiServerMCPClient 加载 MCP 工具

每个 MCP Server 的工具自动打上 _source 属性：
  _source = "mcp:<server_name>"
  例：filesystem server 的工具，_source = "mcp:filesystem"

在 Skill / Agent 配置中引用方式：
  tools:
    - mcp:filesystem     # filesystem server 的全部工具
    - mcp:git            # git server 的全部工具
    - web_search         # 单个工具名
"""
from __future__ import annotations
import asyncio
import logging

logger = logging.getLogger(__name__)


def _tag_mcp_source(tools: list, server_name: str) -> list:
    """给 MCP 工具附加 _source = 'mcp:<server_name>'"""
    source = f"mcp:{server_name}"
    for t in tools:
        try:
            t._source = source
        except AttributeError:
            pass
    return tools


def load_mcp_tools_sync(mcp_config) -> list:
    """
    使用 MultiServerMCPClient 加载所有 MCP Server 工具。
    每次工具调用时自动创建和销毁 session，不存在 session 关闭问题。
    每个 Server 的工具标记独立的 _source。
    """
    if not mcp_config.servers:
        return []

    async def _load():
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client_config = {}
        for name, cfg in mcp_config.servers.items():
            client_config[name] = {
                "transport": "stdio",
                "command": cfg["command"],
                "args": cfg.get("args", []),
                "env": cfg.get("env") or None,
            }

        client = MultiServerMCPClient(client_config)
        # get_tools() 返回所有 server 的工具混合列表，无法区分来源
        # 改为逐个 server 加载，保留来源信息
        all_tools = []
        for name, cfg in client_config.items():
            try:
                single_client = MultiServerMCPClient({name: cfg})
                tools = await single_client.get_tools()
                _tag_mcp_source(tools, name)
                all_tools.extend(tools)
                logger.info(f"MCP [{name}]: 加载了 {len(tools)} 个工具")
            except Exception as e:
                logger.warning(f"MCP [{name}] 加载失败: {e}")

        logger.info(f"MCP 工具总计: {len(all_tools)} 个")
        return all_tools

    try:
        return asyncio.run(_load())
    except Exception as e:
        logger.warning(f"MCP 工具加载失败: {e}")
        return []
