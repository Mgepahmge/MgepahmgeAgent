"""
industrial_db.py - 自定义 MCP Server 示例：工业数据库
将此文件按需修改，然后在 config/mcp_servers.json 中注册：

  "industrial_db": {
    "command": "python",
    "args": ["./mcp_servers/industrial_db.py"]
  }
"""
import os
import json
import psycopg2
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("工业数据库")

# 从环境变量读取连接信息（不要硬编码密码）
_DSN = os.getenv("INDUSTRIAL_DB_DSN", "postgresql://user:pass@localhost:5432/industrial")


def _conn():
    return psycopg2.connect(_DSN)


@mcp.tool()
def query_device_status(device_id: str) -> str:
    """查询指定设备的实时状态"""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT device_id, status, temperature, last_update "
            "FROM devices WHERE device_id = %s",
            (device_id,),
        )
        row = cur.fetchone()
        if not row:
            return f"设备 {device_id} 不存在"
        cols = [d[0] for d in cur.description]
        return json.dumps(dict(zip(cols, row)), default=str, ensure_ascii=False)


@mcp.tool()
def get_alarm_history(device_id: str, hours: int = 24) -> str:
    """获取设备最近 N 小时的报警记录"""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT alarm_code, severity, message, created_at "
            "FROM alarms "
            "WHERE device_id = %s AND created_at > NOW() - INTERVAL '%s hours' "
            "ORDER BY created_at DESC LIMIT 50",
            (device_id, hours),
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return json.dumps(
            [dict(zip(cols, r)) for r in rows],
            default=str,
            ensure_ascii=False,
        )


@mcp.tool()
def get_slow_queries(threshold_ms: int = 1000, limit: int = 20) -> str:
    """从 pg_stat_statements 获取慢查询列表（需要在目标库启用该扩展）"""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT query,
                   round(mean_exec_time::numeric, 2) AS avg_ms,
                   calls,
                   round(total_exec_time::numeric, 2) AS total_ms
            FROM pg_stat_statements
            WHERE mean_exec_time > %s
            ORDER BY mean_exec_time DESC
            LIMIT %s
            """,
            (threshold_ms, limit),
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return json.dumps(
            [dict(zip(cols, r)) for r in rows],
            default=str,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    mcp.run()
