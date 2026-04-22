"""
task_runner.py - 后台任务队列
支持提交长时间运行的任务在后台执行，不阻塞交互。
"""
from __future__ import annotations
import asyncio
import logging
import threading
import time
from .database import create_task, update_task, get_task, list_tasks

logger = logging.getLogger(__name__)

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop, _thread
    if _loop and _loop.is_running():
        return _loop
    _loop = asyncio.new_event_loop()

    def _run(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    _thread = threading.Thread(target=_run, args=(_loop,), daemon=True)
    _thread.start()
    return _loop


def submit_task(description: str, agent, workspace: str) -> str:
    """
    提交一个后台任务，立即返回任务 ID。
    任务在独立事件循环里异步执行。
    """
    tid = create_task(description)
    loop = _get_loop()

    async def _run():
        update_task(tid, status="running", started_at=time.time())
        try:
            from langchain_core.messages import HumanMessage
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=description)], "workspace": workspace},
                config={"configurable": {"thread_id": f"task-{tid}"}},
            )
            last = result["messages"][-1]
            output = last.content if hasattr(last, "content") else str(last)
            update_task(tid, status="done", result=output[:4000], finished_at=time.time())
            logger.info(f"任务 {tid} 完成")
        except Exception as e:
            update_task(tid, status="error", error=str(e), finished_at=time.time())
            logger.error(f"任务 {tid} 失败: {e}")

    asyncio.run_coroutine_threadsafe(_run(), loop)
    return tid


def get_task_status(tid: str) -> dict | None:
    return get_task(tid)


def list_all_tasks(limit: int = 20) -> list[dict]:
    return list_tasks(limit)
