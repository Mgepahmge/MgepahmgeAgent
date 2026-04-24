"""
memory_tools.py - 长期记忆工具插件
Agent 可调用这些工具主动管理长期记忆。

设计：
  - memory_save   : 保存记忆，默认全局，可指定为 agent 私有
  - memory_delete : 按键名删除
  - memory_list   : 列出记忆（含 scope 信息）

符合插件规范：模块级别定义 TOOLS 列表，由 plugin_loader 自动发现。
"""
from langchain_core.tools import tool
from core.database import save_memory, delete_memory_by_key, load_all_memories


@tool
def memory_save(key: str, value: str, scope: str = "global") -> str:
    """
    保存一条长期记忆。
    key   : 记忆的唯一标识（简短描述性名称）
    value : 具体内容
    scope : 'global'（默认，所有 Agent 共享）或 'agent'（当前 Agent 私有）
    同一 key + scope 组合重复保存时覆盖旧值。
    示例：memory_save(key="用户编程语言偏好", value="偏好Python，不喜欢Java")
    """
    try:
        save_memory(key, value, source="agent", scope=scope, agent_id="")
        scope_label = "全局" if scope == "global" else "Agent私有"
        return f"已保存{scope_label}记忆：{key} = {value}"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def memory_delete(key: str, scope: str = "global") -> str:
    """
    删除一条长期记忆。
    key   : 记忆的键名
    scope : 'global' 或 'agent'，需与保存时一致
    示例：memory_delete(key="用户编程语言偏好")
    """
    try:
        delete_memory_by_key(key, scope=scope, agent_id="")
        return f"已删除记忆：{key}（scope={scope}）"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def memory_list() -> str:
    """
    列出所有可见的长期记忆（全局 + 当前 Agent 私有）。
    当用户询问"你记得什么"或需要查看已存储信息时调用。
    """
    memories = load_all_memories(agent_id="")
    if not memories:
        return "当前没有任何长期记忆。"
    lines = []
    for m in memories:
        scope_label = "全局" if m["scope"] == "global" else "私有"
        lines.append(
            f"[{scope_label}] {m['key']}: {m['value']} "
            f"(来源: {m['source'] or '未知'})"
        )
    return "当前长期记忆：\n" + "\n".join(lines)


# 插件规范：必须定义 TOOLS 列表
TOOLS = [memory_save, memory_delete, memory_list]
