"""
project_tools.py - 项目感知工具插件

Agent 可调用这些工具对项目进行深度理解：
  - index_project    : 为指定目录建立索引（自动检测变化）
  - search_code      : 语义检索代码片段
  - get_file_outline : 列出文件中的所有符号
  - find_symbol      : 按名称精确查找符号定义
  - project_status   : 查看当前项目索引状态

符合插件规范：模块级别定义 TOOLS 列表。
"""
from langchain_core.tools import tool


@tool
def index_project(path: str) -> str:
    """
    为指定目录建立深度代码索引（解析函数、类、结构体等符号）。
    支持 Python、C、C++ 代码文件。
    自动检测文件变化：若索引已是最新则直接使用，有变化则重建。
    建立索引后可使用 search_code、find_symbol、get_file_outline 查询。
    示例：index_project(path="/workspace/myproject")
    """
    from core.project_index import ProjectIndex, set_current_index

    try:
        from pathlib import Path
        root = Path(path).expanduser().resolve()
        if not root.exists():
            return f"ERROR: 目录不存在: {path}"
        if not root.is_dir():
            return f"ERROR: 不是目录: {path}"

        index = ProjectIndex(str(root))
        needs, reason = index.needs_reindex()

        if not needs:
            set_current_index(index)
            status = index.status()
            return (
                f"索引已是最新，直接使用。\n"
                f"项目: {root}\n"
                f"符号数: {status['符号数']}，文件数: {status['文件数']}\n"
                f"建立时间: {status['建立时间']}"
            )

        # 需要重建
        lines = [f"开始建立索引: {root}（原因: {reason}）"]
        progress_msgs = []

        def _progress(current, total, msg):
            if current == total or current % 20 == 0:
                progress_msgs.append(f"  [{current}/{total}] {msg}")

        meta = index.build(progress_cb=_progress)
        set_current_index(index)

        lines.append(f"索引建立完成！")
        lines.append(f"  符号数: {meta.symbol_count}")
        lines.append(f"  文件数: {len(meta.file_hashes)}")
        if progress_msgs:
            lines.append("构建过程（摘要）：")
            lines.extend(progress_msgs[-5:])  # 只显示最后5条进度
        return "\n".join(lines)

    except Exception as e:
        import traceback
        return f"ERROR: 索引建立失败: {e}\n{traceback.format_exc()}"


@tool
def search_code(query: str, language: str = "", kind: str = "", k: int = 8) -> str:
    """
    在已索引的项目中语义检索代码片段。
    需要先调用 index_project 建立索引。
    query    : 自然语言描述，例如"数据库连接初始化"、"错误处理逻辑"
    language : 限定语言，可选 python / c / cpp，留空搜索全部
    kind     : 限定符号类型，可选 function / class / struct，留空搜索全部
    k        : 返回结果数量，默认 8
    """
    from core.project_index import get_current_index
    index = get_current_index()
    if index is None:
        return "ERROR: 尚未建立项目索引，请先调用 index_project"

    results = index.search(query, k=k,
                           language=language or None,
                           kind=kind or None)
    if not results:
        return "未找到相关代码片段"

    lines = [f"找到 {len(results)} 个相关代码片段：\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"[{i}] {r['kind']} `{r['name']}` "
            f"({r['language']}) — 相关度 {r['score']:.2f}\n"
            f"    文件: {r['file']}:{r['line']}\n"
            f"    签名: {r['signature']}\n"
        )
        # 展示代码片段（最多 20 行）
        body_lines = r["body"].splitlines()[:20]
        code = "\n".join(f"    {l}" for l in body_lines)
        if len(r["body"].splitlines()) > 20:
            code += "\n    ... (已截断)"
        lines.append(code + "\n")
    return "\n".join(lines)


@tool
def get_file_outline(file_path: str) -> str:
    """
    列出指定文件中的所有符号（函数、类、结构体等）及其行号。
    需要先调用 index_project 建立索引。
    file_path: 文件的绝对路径或相对于项目根目录的路径
    """
    from core.project_index import get_current_index
    index = get_current_index()
    if index is None:
        return "ERROR: 尚未建立项目索引，请先调用 index_project"

    # 尝试补全为绝对路径
    from pathlib import Path
    p = Path(file_path)
    if not p.is_absolute():
        p = index.root / file_path
    abs_path = str(p.resolve())

    symbols = index.get_file_outline(abs_path)
    if not symbols:
        return f"未找到符号（文件可能不在索引中或无符号）: {file_path}"

    lines = [f"文件: {abs_path}，共 {len(symbols)} 个符号：\n"]
    for s in symbols:
        lines.append(f"  第 {s['line']:4d} 行  {s['kind']:10s}  {s['name']}")
        if s.get("signature"):
            lines.append(f"             签名: {s['signature'][:80]}")
    return "\n".join(lines)


@tool
def find_symbol(name: str, kind: str = "") -> str:
    """
    在已索引的项目中精确查找符号定义。
    需要先调用 index_project 建立索引。
    name : 符号名称，例如函数名、类名（大小写敏感）
    kind : 限定类型，可选 function / class / struct，留空搜索全部
    """
    from core.project_index import get_current_index
    index = get_current_index()
    if index is None:
        return "ERROR: 尚未建立项目索引，请先调用 index_project"

    results = index.find_symbol(name, kind=kind or None)
    if not results:
        return f"未找到符号: {name}"

    lines = [f"找到 {len(results)} 处 `{name}` 的定义：\n"]
    for r in results:
        lines.append(
            f"  {r['kind']} `{r['name']}` ({r['language']})\n"
            f"  文件: {r['file']}:{r['line']}\n"
            f"  签名: {r['signature']}\n"
        )
        body_lines = r["body"].splitlines()[:15]
        code = "\n".join(f"    {l}" for l in body_lines)
        if len(r["body"].splitlines()) > 15:
            code += "\n    ... (已截断)"
        lines.append(code + "\n")
    return "\n".join(lines)


@tool
def project_status() -> str:
    """
    查看当前项目索引的状态（是否已索引、符号数量、文件数等）。
    """
    from core.project_index import get_current_index
    index = get_current_index()
    if index is None:
        return "当前没有激活的项目索引。使用 index_project(path=<目录>) 建立索引。"
    status = index.status()
    return "\n".join(f"  {k}: {v}" for k, v in status.items())


# 插件规范：必须定义 TOOLS 列表
TOOLS = [
    index_project,
    search_code,
    get_file_outline,
    find_symbol,
    project_status,
]
