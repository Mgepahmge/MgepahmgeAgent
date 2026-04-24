"""
database.py - 持久化存储层（SQLite）
负责：对话记录、长期记忆（全局/Agent私有）、后台任务、知识集合
"""
from __future__ import annotations
import sqlite3
import json
import time
import uuid
import os
from pathlib import Path
from typing import Optional

DB_PATH = Path(os.getenv("AGENT_DB", "/root/agent/data/agent.db"))


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """初始化所有表，使用 IF NOT EXISTS 保证幂等"""
    with get_conn() as conn:
        conn.executescript("""
        -- 对话会话表
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL DEFAULT '',
            created_at  REAL NOT NULL,
            updated_at  REAL NOT NULL,
            summary     TEXT NOT NULL DEFAULT ''
        );

        -- 消息记录表
        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            tool_calls  TEXT DEFAULT NULL,
            created_at  REAL NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );

        -- 长期记忆表
        -- scope: 'global'（全局，所有Agent可见）或 'agent'（Agent私有）
        -- agent_id: scope='agent' 时标识所属 Agent，global 时为空字符串
        CREATE TABLE IF NOT EXISTS memories (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            key         TEXT NOT NULL,
            value       TEXT NOT NULL,
            scope       TEXT NOT NULL DEFAULT 'global',
            agent_id    TEXT NOT NULL DEFAULT '',
            source      TEXT NOT NULL DEFAULT '',
            created_at  REAL NOT NULL,
            updated_at  REAL NOT NULL,
            UNIQUE(key, scope, agent_id)
        );

        -- 后台任务表
        CREATE TABLE IF NOT EXISTS tasks (
            id          TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'pending',
            result      TEXT NOT NULL DEFAULT '',
            error       TEXT NOT NULL DEFAULT '',
            created_at  REAL NOT NULL,
            started_at  REAL DEFAULT NULL,
            finished_at REAL DEFAULT NULL,
            thread_id   TEXT DEFAULT NULL
        );

        -- 知识集合表（RAG）
        CREATE TABLE IF NOT EXISTS knowledge_collections (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            doc_count   INTEGER NOT NULL DEFAULT 0,
            created_at  REAL NOT NULL,
            updated_at  REAL NOT NULL
        );
        """)


# ──────────────────────────────────────────
# 会话管理
# ──────────────────────────────────────────

def create_session(name: str = "") -> str:
    sid = str(uuid.uuid4())
    now = time.time()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (id, name, created_at, updated_at) VALUES (?,?,?,?)",
            (sid, name or f"对话 {time.strftime('%m-%d %H:%M')}", now, now)
        )
    return sid


def list_sessions(limit: int = 20) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, name, created_at, updated_at, summary "
            "FROM sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_session(sid: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
    return dict(row) if row else None


def update_session_name(sid: str, name: str):
    with get_conn() as conn:
        conn.execute("UPDATE sessions SET name=? WHERE id=?", (name, sid))


def update_session_summary(sid: str, summary: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET summary=?, updated_at=? WHERE id=?",
            (summary, time.time(), sid)
        )


def touch_session(sid: str):
    with get_conn() as conn:
        conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (time.time(), sid))


def delete_session(sid: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM messages WHERE session_id=?", (sid,))
        conn.execute("DELETE FROM sessions WHERE id=?", (sid,))


# ──────────────────────────────────────────
# 消息管理
# ──────────────────────────────────────────

def save_message(session_id: str, role: str, content: str, tool_calls=None):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls, created_at) "
            "VALUES (?,?,?,?,?)",
            (session_id, role, content,
             json.dumps(tool_calls) if tool_calls else None,
             time.time())
        )
    touch_session(session_id)


def load_messages(session_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content, tool_calls FROM messages "
            "WHERE session_id=? ORDER BY id",
            (session_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def count_messages(session_id: str) -> int:
    with get_conn() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id=?", (session_id,)
        ).fetchone()[0]


# ──────────────────────────────────────────
# 长期记忆
# ──────────────────────────────────────────

def save_memory(key: str, value: str, source: str = "",
                scope: str = "global", agent_id: str = ""):
    """
    保存一条长期记忆。
    scope='global'  : 全局记忆，所有 Agent 均可见
    scope='agent'   : Agent 私有记忆，需指定 agent_id
    同一 (key, scope, agent_id) 组合重复保存时覆盖旧值。
    """
    now = time.time()
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO memories (key, value, scope, agent_id, source, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(key, scope, agent_id)
            DO UPDATE SET value=excluded.value,
                          source=excluded.source,
                          updated_at=excluded.updated_at
        """, (key, value, scope, agent_id, source, now, now))


def load_memories(scope: str = "global", agent_id: str = "") -> list[dict]:
    """
    加载指定 scope 的记忆列表。
    scope='global'            : 加载所有全局记忆
    scope='agent', agent_id=X : 加载指定 Agent 的私有记忆
    """
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, key, value, scope, agent_id, source, updated_at "
            "FROM memories WHERE scope=? AND agent_id=? "
            "ORDER BY updated_at DESC",
            (scope, agent_id)
        ).fetchall()
    return [dict(r) for r in rows]


def load_all_memories(agent_id: str = "") -> list[dict]:
    """
    加载全局记忆 + 指定 Agent 的私有记忆（合并，用于注入 system prompt）。
    """
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, key, value, scope, agent_id, source, updated_at "
            "FROM memories "
            "WHERE scope='global' OR (scope='agent' AND agent_id=?) "
            "ORDER BY scope DESC, updated_at DESC",  # global 排前面
            (agent_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def delete_memory_by_id(memory_id: int):
    """按主键 ID 删除（支持序号操作）"""
    with get_conn() as conn:
        conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))


def delete_memory_by_key(key: str, scope: str = "global", agent_id: str = ""):
    """按 (key, scope, agent_id) 删除"""
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM memories WHERE key=? AND scope=? AND agent_id=?",
            (key, scope, agent_id)
        )


def find_memory_by_identifier(identifier: str,
                               scope: str = "global",
                               agent_id: str = "") -> Optional[dict]:
    """
    按序号（数字）或键名查找记忆。
    序号基于 load_memories 返回的排序（updated_at DESC）。
    """
    if identifier.isdigit():
        rows = load_memories(scope, agent_id)
        idx = int(identifier) - 1
        if 0 <= idx < len(rows):
            return rows[idx]
        return None
    # 键名精确匹配
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, key, value, scope, agent_id, source, updated_at "
            "FROM memories WHERE key=? AND scope=? AND agent_id=?",
            (identifier, scope, agent_id)
        ).fetchone()
    return dict(row) if row else None


# ──────────────────────────────────────────
# 后台任务
# ──────────────────────────────────────────

def create_task(description: str) -> str:
    tid = str(uuid.uuid4())[:8]
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO tasks (id, description, status, created_at) VALUES (?,?,?,?)",
            (tid, description, "pending", time.time())
        )
    return tid


def update_task(tid: str, **kwargs):
    if not kwargs:
        return
    sets = ", ".join(f"{k}=?" for k in kwargs)
    vals = list(kwargs.values()) + [tid]
    with get_conn() as conn:
        conn.execute(f"UPDATE tasks SET {sets} WHERE id=?", vals)


def get_task(tid: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone()
    return dict(row) if row else None


def list_tasks(limit: int = 20) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ──────────────────────────────────────────
# 知识集合管理（RAG）
# ──────────────────────────────────────────

def create_collection(name: str, description: str = "") -> str:
    """创建知识集合，返回固定 UUID（前8位）"""
    cid = str(uuid.uuid4())[:8]
    now = time.time()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO knowledge_collections "
            "(id, name, description, doc_count, created_at, updated_at) "
            "VALUES (?,?,?,0,?,?)",
            (cid, name, description, now, now)
        )
    return cid


def get_collection(cid: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM knowledge_collections WHERE id=?", (cid,)
        ).fetchone()
    return dict(row) if row else None


def list_collections() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM knowledge_collections ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def update_collection_count(cid: str, delta: int):
    with get_conn() as conn:
        conn.execute(
            "UPDATE knowledge_collections "
            "SET doc_count=MAX(0, doc_count+?), updated_at=? WHERE id=?",
            (delta, time.time(), cid)
        )


def delete_collection(cid: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM knowledge_collections WHERE id=?", (cid,))


def find_collection_by_identifier(identifier: str) -> Optional[dict]:
    """按序号（数字）或 ID 前缀查找集合"""
    if identifier.isdigit():
        cols = list_collections()
        idx = int(identifier) - 1
        if 0 <= idx < len(cols):
            return cols[idx]
        return None
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM knowledge_collections WHERE id LIKE ?",
            (identifier + "%",)
        ).fetchone()
    return dict(row) if row else None
