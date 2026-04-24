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

def merge_skills(skill_ids: list[str],
                 registry: SkillRegistry,
                 all_tools: list) -> tuple[str, list, list[str]]:
    """
    将指定 Skill 列表的能力合并，返回：
      (combined_prompt, filtered_tools, combined_knowledge_ids)

    - combined_prompt    : 所有 Skill 的 system_prompt 拼接
    - filtered_tools     : base_tools + Skill 指定工具（去重，保持顺序）
    - combined_knowledge : 所有 Skill 的 knowledge 集合 ID（去重）
    """
    prompt_parts: list[str] = []
    extra_tool_names: set[str] = set()
    knowledge_ids: list[str] = []
    seen_knowledge: set[str] = set()

    for sid in skill_ids:
        skill = registry.get(sid)
        if skill is None:
            logger.warning(f"Skill '{sid}' 不存在，已跳过")
            continue
        if skill.system_prompt:
            prompt_parts.append(f"【{skill.name}】\n{skill.system_prompt}")
        extra_tool_names.update(skill.tools)
        for kid in skill.knowledge:
            if kid not in seen_knowledge:
                knowledge_ids.append(kid)
                seen_knowledge.add(kid)

    # 合并工具：base_tools 中按名称过滤出 Skill 指定的额外工具
    tool_name_map = {t.name: t for t in all_tools}
    merged_tools = list(all_tools)  # 先放全部 base_tools
    for name in extra_tool_names:
        if name not in {t.name for t in merged_tools}:
            if name in tool_name_map:
                merged_tools.append(tool_name_map[name])
            else:
                logger.warning(f"Skill 引用的工具 '{name}' 不存在")

    combined_prompt = "\n\n".join(prompt_parts)
    return combined_prompt, merged_tools, knowledge_ids


# 全局单例
skill_registry = SkillRegistry()
