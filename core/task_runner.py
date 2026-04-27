"""
task_runner.py - 后台任务队列

提交任务后立即返回任务 ID，任务在 AgentRuntime 的独立事件循环里异步执行。
与多 Agent 架构完全兼容：每个任务绑定到提交时的 AgentRuntime。
"""
from __future__ import annotations
import asyncio
import logging
import time
from .database import create_task, update_task, get_task, list_tasks

logger = logging.getLogger(__name__)


def submit_task(description: str, agent_id: str) -> str:
    """
    提交一个后台任务，立即返回任务 ID。
    任务在指定 Agent 的独立事件循环里异步执行，不阻塞交互。

    agent_id: 执行此任务的 Agent ID（对应 AgentRegistry 里的运行实例）
    """
    from core.agent_registry import agent_registry
    runtime = agent_registry.get_runtime(agent_id)
    if runtime is None:
        raise RuntimeError(f"Agent '{agent_id}' 未启动，无法提交任务")

    # 为任务创建独立的 session ID，与交互对话隔离
    task_session_id = f"task-{int(time.time())}"

    tid = create_task(description)
    update_task(tid, status="running", started_at=time.time(), thread_id=task_session_id)

    async def _run():
        try:
            output = await runtime._ainvoke(description, task_session_id)
            update_task(tid, status="done",
                        result=output[:4000], finished_at=time.time())
            logger.info(f"任务 [{tid}] 完成（Agent: {agent_id}）")
        except Exception as e:
            update_task(tid, status="error",
                        error=str(e), finished_at=time.time())
            logger.error(f"任务 [{tid}] 失败: {e}")

    # 在 Agent 自己的事件循环里提交，保持工具和上下文的一致性
    asyncio.run_coroutine_threadsafe(_run(), runtime._loop)
    return tid


def get_task_status(tid: str) -> dict | None:
    return get_task(tid)


def list_all_tasks(limit: int = 20) -> list[dict]:
    return list_tasks(limit)
