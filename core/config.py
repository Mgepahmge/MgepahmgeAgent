"""
config.py - 统一配置加载，支持多 LLM Provider
"""
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

PROVIDERS_FILE = Path(__file__).parent.parent / "config" / "providers.json"


def _expand(val: str) -> str:
    for k, v in os.environ.items():
        val = val.replace(f"${{{k}}}", v)
    return os.path.expanduser(val)


# ──────────────────────────────────────────────────────
# Provider 数据结构
# ──────────────────────────────────────────────────────

@dataclass
class ProviderProfile:
    """一个 LLM provider 配置"""
    name: str                   # 唯一标识，例如 "claude" / "openai" / "deepseek"
    type: str                   # 驱动类型: anthropic | openai | ollama
    api_key: str = ""
    base_url: str = ""          # 自定义 base_url（OpenAI 兼容接口必填）
    model: str = ""
    max_tokens: int = 8192

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.model,
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProviderProfile":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


class ProviderRegistry:
    """
    管理所有 LLM Provider，持久化到 config/providers.json。

    接口：
      registry.list()            → 所有 provider 名称
      registry.get(name)         → ProviderProfile | None
      registry.add(profile)      → 添加/更新并保存
      registry.remove(name)      → 删除
      registry.set_active(name)  → 切换当前 provider
      registry.active()          → 当前 ProviderProfile | None
      registry.active_name()     → 当前名称字符串
    """

    def __init__(self):
        self._profiles: dict[str, ProviderProfile] = {}
        self._active: str = ""
        self._load()

    def _load(self):
        if PROVIDERS_FILE.exists():
            try:
                raw = json.loads(PROVIDERS_FILE.read_text())
                self._active = raw.get("active", "")
                for d in raw.get("providers", []):
                    p = ProviderProfile.from_dict(d)
                    self._profiles[p.name] = p
                return
            except Exception:
                pass
        self._seed_defaults()
        self._save()

    def _save(self):
        PROVIDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "active": self._active,
            "providers": [p.to_dict() for p in self._profiles.values()],
        }
        PROVIDERS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def _seed_defaults(self):
        """首次运行写入默认示例"""
        defaults = [
            ProviderProfile(
                name="claude",
                type="anthropic",
                api_key=os.getenv("ANTHROPIC_API_KEY", ""),
                model="claude-sonnet-4-20250514",
            ),
            ProviderProfile(
                name="openai",
                type="openai",
                api_key=os.getenv("OPENAI_API_KEY", ""),
                base_url="https://api.openai.com/v1",
                model="gpt-4o",
            ),
            ProviderProfile(
                name="deepseek",
                type="openai",          # DeepSeek 兼容 OpenAI SDK
                api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                base_url="https://api.deepseek.com/v1",
                model="deepseek-chat",
            ),
            ProviderProfile(
                name="ollama-local",
                type="ollama",
                base_url="http://localhost:11434",
                model="qwen2.5:14b",
            ),
        ]
        for p in defaults:
            self._profiles[p.name] = p
        # 自动激活第一个有 key 的
        for p in defaults:
            if p.api_key or p.type == "ollama":
                self._active = p.name
                break

    # ── 公开接口 ──────────────────────────────────────

    def list(self) -> list[str]:
        return list(self._profiles.keys())

    def get(self, name: str) -> Optional[ProviderProfile]:
        return self._profiles.get(name)

    def add(self, profile: ProviderProfile):
        self._profiles[profile.name] = profile
        if not self._active:
            self._active = profile.name
        self._save()

    def remove(self, name: str) -> bool:
        if name not in self._profiles:
            return False
        del self._profiles[name]
        if self._active == name:
            self._active = next(iter(self._profiles), "")
        self._save()
        return True

    def set_active(self, name: str) -> bool:
        if name not in self._profiles:
            return False
        self._active = name
        self._save()
        return True

    def active(self) -> Optional[ProviderProfile]:
        return self._profiles.get(self._active)

    def active_name(self) -> str:
        return self._active


# ──────────────────────────────────────────────────────
# RAG / MCP / Agent 配置
# ──────────────────────────────────────────────────────

@dataclass
class RAGConfig:
    enabled: bool = False
    host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", ""))
    port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    db: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "agent_kb"))
    user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", ""))
    password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", ""))
    table: str = field(default_factory=lambda: os.getenv("VECTOR_TABLE", "documents"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    embedding_device: str = field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cuda"))

    @property
    def connection_string(self) -> str:
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"

    def __post_init__(self):
        self.enabled = bool(self.host and self.user and self.password)


@dataclass
class MCPConfig:
    config_path: str = field(default_factory=lambda: os.getenv("MCP_CONFIG_PATH", "./config/mcp_servers.json"))
    servers: dict = field(default_factory=dict)

    def __post_init__(self):
        path = Path(self.config_path)
        if path.exists():
            raw = json.loads(path.read_text())
            for name, cfg in raw.get("servers", {}).items():
                cfg["args"] = [_expand(a) for a in cfg.get("args", [])]
                self.servers[name] = cfg


@dataclass
class AgentConfig:
    workspace_dir: str = field(default_factory=lambda: _expand(os.getenv("WORKSPACE_DIR", "~/workspace")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_file: str = field(default_factory=lambda: os.getenv("LOG_FILE", "./logs/agent.log"))
    rag: RAGConfig = field(default_factory=RAGConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    providers: ProviderRegistry = field(default_factory=ProviderRegistry)


# 全局单例
config = AgentConfig()
