"""
mcp_loader.py - 使用 MultiServerMCPClient 加载 MCP 工具
MultiServerMCPClient 默认为每次工具调用自动创建新 session，无需手动管理 session 生命周期。
"""
from __future__ import annotations
import asyncio
import logging

logger = logging.getLogger(__name__)


def load_mcp_tools_sync(mcp_config) -> list:
    """
    使用 MultiServerMCPClient 加载所有 MCP Server 工具。
    每次工具调用时自动创建和销毁 session，不存在 session 关闭问题。
    """
    if not mcp_config.servers:
        return []

    async def _load():
        from langchain_mcp_adapters.client import MultiServerMCPClient

        # 将配置转换为 MultiServerMCPClient 所需格式
        client_config = {}
        for name, cfg in mcp_config.servers.items():
            client_config[name] = {
                "transport": "stdio",
                "command": cfg["command"],
                "args": cfg.get("args", []),
                "env": cfg.get("env") or None,
            }

        client = MultiServerMCPClient(client_config)
        tools = await client.get_tools()
        logger.info(f"MCP 工具总计: {len(tools)} 个")
        return tools

    try:
        return asyncio.run(_load())
    except Exception as e:
        logger.warning(f"MCP 工具加载失败: {e}")
        return []
