# CLI Agent 部署指南

## 目录结构

```
agent/
├── cli.py                    # 入口，直接运行这个
├── requirements.txt
├── .env.example              # 复制为 .env 并填写
├── core/
│   ├── config.py             # 配置加载
│   └── agent_graph.py        # LangGraph 图定义（Agent 核心）
├── tools/
│   ├── builtin_tools.py      # 内置工具：文件/Web/Shell
│   └── mcp_loader.py         # MCP Server 动态加载
├── rag/
│   └── knowledge_base.py     # pgvector RAG
├── mcp_servers/
│   └── industrial_db.py      # 自定义 MCP Server 示例
└── config/
    └── mcp_servers.json      # MCP 服务配置
```

---

## 第一步：系统依赖

```bash
# Python 3.11+
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip \
                   nodejs npm uvx \
                   libpq-dev build-essential

# 验证 CUDA（如已安装驱动）
nvidia-smi
nvcc --version
```

---

## 第二步：创建 Python 虚拟环境

```bash
cd ~/agent
python3.11 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 如果要使用 CUDA 加速 Embedding（PyTorch CUDA 版本）
# 先去 https://pytorch.org 选对应 CUDA 版本的安装命令，例如：
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 第三步：配置

```bash
cp .env.example .env
vim .env   # 填写 API Key 和数据库信息
```

**最小配置（不用 RAG）：**
```env
ANTHROPIC_API_KEY=sk-ant-...
WORKSPACE_DIR=~/workspace
```

**完整配置（启用 RAG）：**
```env
ANTHROPIC_API_KEY=sk-ant-...
WORKSPACE_DIR=~/workspace

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=agent_kb
POSTGRES_USER=agent
POSTGRES_PASSWORD=your_password

EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cuda
```

---

## 第四步：（可选）搭建 pgvector 知识库

```bash
# 安装 PostgreSQL + pgvector 扩展
sudo apt install -y postgresql postgresql-contrib
sudo -u postgres psql <<EOF
CREATE USER agent WITH PASSWORD 'your_password';
CREATE DATABASE agent_kb OWNER agent;
\c agent_kb
CREATE EXTENSION vector;
GRANT ALL ON SCHEMA public TO agent;
EOF

# 验证
psql -U agent -d agent_kb -c "SELECT extname FROM pg_extension;"
# 应该能看到 vector
```

---

## 第五步：运行

```bash
# 每次使用前激活虚拟环境
source ~/agent/.venv/bin/activate

# 进入交互式对话
python cli.py

# 单次提问
python cli.py "帮我列出 workspace 里所有 Python 文件"

# 导入文档到知识库
python cli.py ingest ~/documents/manuals

# 查看所有可用工具
python cli.py tools
```

---

## 第六步：做成全局命令（可选）

```bash
# 创建启动脚本
cat > ~/.local/bin/agent << 'EOF'
#!/bin/bash
source ~/agent/.venv/bin/activate
python ~/agent/cli.py "$@"
EOF
chmod +x ~/.local/bin/agent

# 确保 ~/.local/bin 在 PATH 里
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 之后直接用：
agent
agent "帮我分析这个日志"
agent ingest ./docs
```

---

## 添加自定义 MCP Server

### 1. 写 Server 代码
参考 `mcp_servers/industrial_db.py`，用 FastMCP 框架：

```python
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("我的服务")

@mcp.tool()
def my_tool(param: str) -> str:
    """工具描述，Agent 会读这段描述来决定何时调用"""
    return "结果"

if __name__ == "__main__":
    mcp.run()
```

### 2. 在配置里注册
编辑 `config/mcp_servers.json`：

```json
{
  "servers": {
    "my_service": {
      "command": "python",
      "args": ["./mcp_servers/my_service.py"],
      "description": "我的自定义服务"
    }
  }
}
```

### 3. 重启 Agent 即可，新工具自动加载

---

## 导入文档到 RAG

```bash
# 支持格式：PDF / TXT / MD / PY / CPP / JAVA / DOCX / JSON / YAML / SQL
agent ingest ~/manuals/              # 整个目录
agent ingest ~/manuals/device.pdf    # 单个文件

# 交互模式中也可以用斜杠命令：
# /ingest ~/manuals/
```

---

## 交互模式斜杠指令

| 指令 | 功能 |
|------|------|
| `/help` | 显示帮助 |
| `/tools` | 列出所有可用工具 |
| `/ingest <路径>` | 导入文档到知识库 |
| `/clear` | 清空对话记忆，开新会话 |
| `/quit` | 退出 |

---

## 升级到本地模型（无需 API 费用）

```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 下载模型（选一个，越大效果越好，需要 VRAM 够）
ollama pull qwen2.5:14b      # 14B，约需 10GB VRAM
ollama pull qwen2.5:32b      # 32B，约需 20GB VRAM
ollama pull deepseek-r1:14b  # 擅长推理

# 修改 .env
OLLAMA_BASE_URL=http://localhost:11434
LOCAL_MODEL=qwen2.5:14b
# 注释掉 ANTHROPIC_API_KEY 或留空
```

---

## 常见问题

**Q: CUDA Embedding 报错**
```bash
# 检查 CUDA 是否可用
python -c "import torch; print(torch.cuda.is_available())"
# 如果返回 False，在 .env 中设置：
EMBEDDING_DEVICE=cpu
```

**Q: MCP Server 加载失败**
```bash
# 检查 npx / uvx 是否安装
npx --version
uvx --version   # pip install uv

# 手动测试 MCP Server
python mcp_servers/industrial_db.py
```

**Q: pgvector 连接失败**
```bash
# 测试连接
psql -U agent -d agent_kb -h localhost
# 检查扩展
\dx
```

**Q: 找不到 agent 命令**
```bash
source ~/.bashrc
which agent
```
