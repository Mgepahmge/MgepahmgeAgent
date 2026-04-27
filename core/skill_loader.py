"""
skill_loader.py - Skill 加载和合并逻辑

Skill 定义在 skills/ 目录下，每个 .yaml 文件一个 Skill。
文件名（不含扩展名）为 Skill 的唯一 ID。

Skill 结构：
  name          : 显示名称
  description   : 简短描述
  system_prompt : 追加到 Agent system prompt 末尾的片段
  tools         : 额外工具名称列表（从已注册工具中按名称选取）
  knowledge     : 关联的知识集合 ID 列表（用于 RAG 过滤检索）
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).parent.parent / "skills"


@dataclass
class Skill:
    """一个已加载的 Skill"""
    id: str                          # 文件名（不含扩展名），全局唯一
    name: str
    description: str = ""
    system_prompt: str = ""
    tools: list[str] = field(default_factory=list)
    knowledge: list[str] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: Path) -> "Skill":
        """从 YAML 文件加载一个 Skill，校验必要字段"""
        if yaml is None:
            raise ImportError("PyYAML 未安装，请运行：pip install pyyaml")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        sid = path.stem
        name = data.get("name", "").strip()
        if not name:
            raise ValueError(f"Skill '{sid}' 缺少必填字段 'name'")
        return cls(
            id=sid,
            name=name,
            description=str(data.get("description", "")).strip(),
            system_prompt=str(data.get("system_prompt", "")).strip(),
            tools=[str(t).strip() for t in (data.get("tools") or [])],
            knowledge=[str(k).strip() for k in (data.get("knowledge") or [])],
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "tools": self.tools,
            "knowledge": self.knowledge,
        }


class SkillRegistry:
    """
    管理所有已发现的 Skill。
    启动时扫描 skills/ 目录，按需加载。

    接口：
      registry.all()              → 所有 Skill 列表
      registry.get(sid)           → Skill | None
      registry.reload()           → 重新扫描目录
    """

    def __init__(self):
        self._skills: dict[str, Skill] = {}
        self.reload()

    def reload(self):
        """扫描 skills/ 目录，加载所有 .yaml 文件"""
        self._skills.clear()
        if not SKILLS_DIR.exists():
            SKILLS_DIR.mkdir(parents=True)
            logger.info("已创建 skills/ 目录")
            return
        for path in sorted(SKILLS_DIR.glob("*.yaml")):
            if path.stem.startswith("_") or path.stem == "example":
                continue  # 跳过私有文件和示例文件
            try:
                skill = Skill.from_file(path)
                self._skills[skill.id] = skill
                logger.info(f"Skill [{skill.id}]: {skill.name} 加载成功")
            except Exception as e:
                logger.warning(f"Skill [{path.stem}] 加载失败: {e}")
        logger.info(f"共加载 {len(self._skills)} 个 Skill")

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    def get(self, sid: str) -> Optional[Skill]:
        return self._skills.get(sid)

    def exists(self, sid: str) -> bool:
        return sid in self._skills


# ──────────────────────────────────────────
# Skill 合并：将多个 Skill 的能力合并到 Agent
# ──────────────────────────────────────────

def resolve_tool_refs(refs: list[str], all_tools: list) -> list:
    """
    将工具引用列表解析为实际工具对象（去重，保持顺序）。

    引用格式：
      "web_search"       → 按工具名精确匹配
      "memory_tools"     → 按 _source 匹配（插件文件名）
      "mcp:filesystem"   → 按 _source 匹配（MCP server）
      "builtin"          → 按 _source 匹配（内置工具）

    空列表表示使用全部工具。
    """
    if not refs:
        return list(all_tools)

    resolved: list = []
    seen_names: set[str] = set()

    def _add(tool):
        if tool.name not in seen_names:
            seen_names.add(tool.name)
            resolved.append(tool)

    # 建立快速查找索引
    by_name: dict[str, object] = {t.name: t for t in all_tools}
    by_source: dict[str, list] = {}
    for t in all_tools:
        src = getattr(t, "_source", "")
        if src:
            by_source.setdefault(src, []).append(t)

    for ref in refs:
        ref = ref.strip()
        if not ref:
            continue
        if ref in by_name:
            # 精确工具名匹配
            _add(by_name[ref])
        elif ref in by_source:
            # source 批量匹配（插件名 或 mcp:server）
            for t in by_source[ref]:
                _add(t)
        else:
            logger.warning(f"工具引用 '{ref}' 未匹配任何工具或来源，已跳过")

    return resolved


def merge_skills(skill_ids: list[str],
                 registry: SkillRegistry,
                 base_tools: list,
                 full_tool_pool: list | None = None) -> tuple[str, list, list[str]]:
    """
    将指定 Skill 列表的能力合并，返回：
      (combined_prompt, merged_tools, combined_knowledge_ids)

    - combined_prompt  : 所有 Skill 的 system_prompt 拼接
    - merged_tools     : base_tools + Skill 额外工具（去重）
    - knowledge_ids    : 所有 Skill 的 knowledge 集合 ID（去重）

    base_tools     : Agent 的基础工具集（已按 base_tools 配置过滤）
    full_tool_pool : 全量注册工具，Skill.tools 从此范围内解析
                     None 时回退到 base_tools（向后兼容）
    """
    prompt_parts: list[str] = []
    extra_refs: list[str] = []
    knowledge_ids: list[str] = []
    seen_knowledge: set[str] = set()

    for sid in skill_ids:
        skill = registry.get(sid)
        if skill is None:
            logger.warning(f"Skill '{sid}' 不存在，已跳过")
            continue
        if skill.system_prompt:
            prompt_parts.append(f"【{skill.name}】\n{skill.system_prompt}")
        extra_refs.extend(skill.tools)
        for kid in skill.knowledge:
            if kid not in seen_knowledge:
                knowledge_ids.append(kid)
                seen_knowledge.add(kid)

    # 合并工具：在 base_tools 基础上追加 Skill 额外指定的（去重）
    # Skill 工具从 full_tool_pool 里解析，确保能找到不在 base_tools 里的工具
    pool = full_tool_pool if full_tool_pool is not None else base_tools
    seen = {t.name for t in base_tools}
    extra_tools = [
        t for t in resolve_tool_refs(extra_refs, pool)
        if t.name not in seen
    ]
    merged_tools = list(base_tools) + extra_tools

    combined_prompt = "\n\n".join(prompt_parts)
    return combined_prompt, merged_tools, knowledge_ids


# 全局单例
skill_registry = SkillRegistry()
