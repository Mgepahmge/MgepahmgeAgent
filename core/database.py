"""
database.py - 持久化存储层（SQLite）
负责：对话记录、长期记忆、后台任务
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
    """初始化所有表"""
    with get_conn() as conn:
        conn.executescript("""
        -- 对话会话表
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            name TEXT,
            created_at REAL,
            updated_at REAL,
            summary TEXT DEFAULT ''
        );

        -- 消息记录表
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tool_calls TEXT DEFAULT NULL,
            created_at REAL,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        );

        -- 长期记忆表
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE NOT NULL,
            value TEXT NOT NULL,
            source TEXT DEFAULT '',
            created_at REAL,
            updated_at REAL
        );

        -- 后台任务表
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            result TEXT DEFAULT '',
            error TEXT DEFAULT '',
            created_at REAL,
            started_at REAL DEFAULT NULL,
            finished_at REAL DEFAULT NULL,
            thread_id TEXT DEFAULT NULL
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
            "SELECT id, name, created_at, updated_at, summary FROM sessions "
            "ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_session(sid: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
    return dict(row) if row else None


def update_session_summary(sid: str, summary: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET summary=?, updated_at=? WHERE id=?",
            (summary, time.time(), sid)
        )


def touch_session(sid: str):
    with get_conn() as conn:
        conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (time.time(), sid))


# ──────────────────────────────────────────
# 消息管理
# ──────────────────────────────────────────

def save_message(session_id: str, role: str, content: str, tool_calls=None):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls, created_at) VALUES (?,?,?,?,?)",
            (session_id, role, content, json.dumps(tool_calls) if tool_calls else None, time.time())
        )
    touch_session(session_id)


def load_messages(session_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content, tool_calls FROM messages WHERE session_id=? ORDER BY id",
            (session_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def count_messages(session_id: str) -> int:
    with get_conn() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id=?", (session_id,)
        ).fetchone()[0]


def delete_session(sid: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM messages WHERE session_id=?", (sid,))
        conn.execute("DELETE FROM sessions WHERE id=?", (sid,))


# ──────────────────────────────────────────
# 长期记忆
# ──────────────────────────────────────────

def save_memory(key: str, value: str, source: str = ""):
    now = time.time()
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO memories (key, value, source, created_at, updated_at)
            VALUES (?,?,?,?,?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value,
                source=excluded.source, updated_at=excluded.updated_at
        """, (key, value, source, now, now))


def load_all_memories() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT key, value, source, updated_at FROM memories ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_memory(key: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM memories WHERE key=?", (key,))


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
