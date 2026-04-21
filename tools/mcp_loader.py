"""
mcp_loader.py - 动态加载 MCP Server 并转换为 LangChain 工具
"""
from __future__ import annotations
import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def _load_one_server(name: str, cfg: dict) -> list:
    """启动单个 MCP Server，返回对应的 LangChain 工具列表"""
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from langchain_mcp_adapters.tools import load_mcp_tools

        params = StdioServerParameters(
            command=cfg["command"],
            args=cfg.get("args", []),
            env=cfg.get("env"),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                logger.info(f"MCP [{name}]: 加载了 {len(tools)} 个工具")
                return tools
    except Exception as e:
        logger.warning(f"MCP [{name}] 加载失败: {e}")
        return []


def load_mcp_tools_sync(mcp_config) -> list:
    """
    同步入口：加载所有配置的 MCP Server 工具。
    失败的 Server 会被跳过，不影响其他工具。
    """
    if not mcp_config.servers:
        return []

    all_tools = []
    for name, cfg in mcp_config.servers.items():
        try:
            tools = asyncio.run(_load_one_server(name, cfg))
            all_tools.extend(tools)
        except Exception as e:
            logger.warning(f"MCP [{name}] 跳过: {e}")

    logger.info(f"MCP 工具总计: {len(all_tools)} 个")
    return all_tools
