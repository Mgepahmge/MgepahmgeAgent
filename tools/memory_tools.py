"""
memory_tools.py - 长期记忆工具插件
Agent 可调用这些工具主动管理长期记忆。
符合插件规范：模块级别定义 TOOLS 列表，由 plugin_loader 自动发现。
"""
from langchain_core.tools import tool
from core.database import save_memory, delete_memory, load_all_memories


@tool
def memory_save(key: str, value: str) -> str:
    """
    保存一条长期记忆。key 是记忆的唯一标识，value 是具体内容。
    当用户要求记住某件事时调用此工具。
    同一个 key 再次保存会覆盖旧值。
    示例：memory_save(key="用户编程语言偏好", value="偏好Python，不喜欢Java")
    """
    try:
        save_memory(key, value, source="agent")
        return f"已保存记忆：{key} = {value}"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def memory_delete(key: str) -> str:
    """
    删除一条长期记忆。
    当用户要求忘记某件事时调用此工具。
    示例：memory_delete(key="用户编程语言偏好")
    """
    try:
        delete_memory(key)
        return f"已删除记忆：{key}"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def memory_list() -> str:
    """
    列出所有长期记忆。
    当需要查看已记住的信息，或用户询问"你记得什么"时调用。
    """
    memories = load_all_memories()
    if not memories:
        return "当前没有任何长期记忆。"
    lines = [
        f"- {m['key']}: {m['value']} (来源: {m['source'] or '未知'})"
        for m in memories
    ]
    return "当前长期记忆：\n" + "\n".join(lines)


# 插件规范：必须定义 TOOLS 列表
TOOLS = [memory_save, memory_delete, memory_list]
