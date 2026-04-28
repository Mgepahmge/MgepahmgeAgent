"""
config.py - 统一配置加载，支持多 LLM Provider
API Key 通过环境变量引用，不存明文
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
    """
    一个 LLM provider 配置。
    api_key      : 直接存明文（仅用于手动编辑/测试，不推荐）
    api_key_env  : 推荐方式，填环境变量名，运行时动态读取
                   例如 "PROVIDER_KEY_CLAUDE"
    resolved_api_key 属性统一读取最终 key。
    """
    name: str
    type: str                   # anthropic | openai | ollama
    api_key: str = ""           # 明文（不推荐）
    api_key_env: str = ""       # 环境变量名（推荐）
    base_url: str = ""
    model: str = ""
    max_tokens: int = 8192

    @property
    def resolved_api_key(self) -> str:
        """优先从环境变量读取，其次用字段明文值"""
        if self.api_key_env:
            val = os.getenv(self.api_key_env, "")
            if val:
                return val
        return self.api_key

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type,
            "api_key": "",              # 序列化时永远不写明文 key
            "api_key_env": self.api_key_env,
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
    文件中只存 api_key_env（变量名），不存明文 Key。
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
        defaults = [
            ProviderProfile(
                name="claude",
                type="anthropic",
                api_key_env="PROVIDER_KEY_CLAUDE",
                model="claude-sonnet-4-20250514",
            ),
            ProviderProfile(
                name="openai",
                type="openai",
                api_key_env="PROVIDER_KEY_OPENAI",
                base_url="https://api.openai.com/v1",
                model="gpt-4o",
            ),
            ProviderProfile(
                name="deepseek",
                type="openai",
                api_key_env="PROVIDER_KEY_DEEPSEEK",
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
            if p.resolved_api_key or p.type == "ollama":
                self._active = p.name
                break
        if not self._active and defaults:
            self._active = defaults[0].name

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
                # env 字段：展开 ${VAR} 占位符，值和键都支持
                if cfg.get("env") and isinstance(cfg["env"], dict):
                    cfg["env"] = {
                        _expand(k): _expand(v)
                        for k, v in cfg["env"].items()
                    }
                self.servers[name] = cfg


@dataclass
class AgentConfig:
    workspace_dir: str = field(default_factory=lambda: _expand(os.getenv("WORKSPACE_DIR", "/workspace")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_file: str = field(default_factory=lambda: os.getenv("LOG_FILE", "./logs/agent.log"))
    rag: RAGConfig = field(default_factory=RAGConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    providers: ProviderRegistry = field(default_factory=ProviderRegistry)


# 全局单例
config = AgentConfig()
