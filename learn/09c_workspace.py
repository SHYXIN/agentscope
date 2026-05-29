# -*- coding: utf-8 -*-
"""
实战 9c: Workspace — Agent 的工作空间
=======================================
Workspace 为 Agent 提供持久化的工作环境:
  - 文件存储 (data/ 目录)
  - 技能管理 (skills/ 目录)
  - MCP 客户端配置 (.mcp 文件)
  - 会话数据 (sessions/ 目录)

本实战演示 LocalWorkspace (本地目录作为工作空间)

运行方式:
    .venv\Scripts\python.exe learn\09c_workspace.py
"""

import os
import sys
import asyncio
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from agentscope.workspace import LocalWorkspace
from agentscope.skill import LocalSkillLoader


# ============================================================
# 第一步: 创建 Skill 目录 (供 Workspace 加载)
# ============================================================
def setup_skills(workspace_dir: Path):
    """在 Workspace 的 skills 目录中创建 Skill"""
    skill_dir = workspace_dir / "skills" / "writer"
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("""---
name: Technical Writer
description: 技术写作技能，帮助撰写清晰、专业的技术文档
---

# 技术写作技能

你是一个专业的技术写手。请遵循以下原则:

## 写作原则
1. 清晰 — 用简单的语言解释复杂概念
2. 结构化 — 使用标题、列表、表格组织内容
3. 准确 — 确保技术细节正确
4. 完整 — 覆盖所有重要方面

## 文档结构
- 引言: 概述主题
- 正文: 详细解释 (分多个小节)
- 总结: 关键要点
- 参考: 相关资源 (可选)
""", encoding="utf-8")

    print(f"[Workspace] 已创建 Skill: {skill_dir}")


# ============================================================
# 第二步: 初始化 Workspace
# ============================================================
async def main():
    print("=" * 60)
    print("Workspace — Agent 的工作空间")
    print("=" * 60)

    # 1. 创建工作空间目录
    workspace_dir = Path(__file__).parent / "workspace_demo"
    workspace_dir.mkdir(exist_ok=True)
    print(f"\n[Workspace] 工作目录: {workspace_dir}")

    # 2. 创建 Skill
    setup_skills(workspace_dir)

    # 3. 初始化 LocalWorkspace
    workspace = LocalWorkspace(workdir=str(workspace_dir))
    await workspace.initialize()

    print(f"[Workspace ID] {workspace.workspace_id}")
    print(f"[Workspace 状态] is_alive={workspace.is_alive}")

    # 4. 获取工作空间指令
    instructions = await workspace.get_instructions()
    print(f"\n[工作空间指令]\n{instructions}")

    # 5. 列出已加载的技能
    skills = await workspace.list_skills()
    print(f"\n[已加载技能] {len(skills)} 个")
    for skill in skills:
        print(f"  - {skill.name}: {skill.description}")
        print(f"    目录: {skill.dir}")

    # 6. 列出 MCP 客户端
    mcps = await workspace.list_mcps()
    print(f"\n[MCP 客户端] {len(mcps)} 个")
    for mcp in mcps:
        print(f"  - {mcp.name}")

    # 7. 列出工具
    tools = await workspace.list_tools()
    print(f"\n[工具] {len(tools)} 个")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:50]}...")

    # 8. 查看工作空间目录结构
    print(f"\n[工作空间目录结构]")
    for root, dirs, files in os.walk(workspace_dir):
        level = root.replace(str(workspace_dir), "").count(os.sep)
        indent = "  " * level
        print(f"  {indent}{os.path.basename(root)}/")
        for file in files[:5]:  # 最多显示 5 个文件
            print(f"  {indent}  {file}")

    # 9. 关闭工作空间
    await workspace.close()
    print(f"\n[Workspace 已关闭] is_alive={workspace.is_alive}")

    # ========================================
    # 总结
    # ========================================
    print("\n" + "=" * 60)
    print("[Workspace 总结]")
    print("=" * 60)
    print("""
Workspace 核心功能:

1. 目录管理
   - workdir/: 工作空间根目录
   - data/    : 存储大型文件 (图片、文档等)
   - skills/  : 技能目录 (每个子目录是一个 Skill)
   - sessions/: 会话数据 (上下文、工具结果)
   - .mcp     : MCP 客户端配置 (JSON)

2. 技能加载
   - 自动从 skills/ 目录加载 Skill
   - 每个 Skill 是一个包含 SKILL.md 的子目录
   - SKILL.md 使用 frontmatter 定义 name/description

3. MCP 管理
   - MCP 客户端配置持久化到 .mcp 文件
   - 重启后自动恢复 MCP 连接

4. 上下文卸载 (Offload)
   - 大型工具结果可以卸载到磁盘
   - 节省 context window 空间
   - 支持图片、文档等多媒体文件

使用方式:
  workspace = LocalWorkspace(workdir="/path/to/workspace")
  await workspace.initialize()
  skills = await workspace.list_skills()
  mcps = await workspace.list_mcps()
  await workspace.close()
    """)


if __name__ == "__main__":
    asyncio.run(main())
