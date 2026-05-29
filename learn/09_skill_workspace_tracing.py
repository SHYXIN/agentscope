# -*- coding: utf-8 -*-
"""
实战 9: Skill + Workspace + Tracing
=====================================
演示 AgentScope 的三大高级功能:
  1. Skill — 可复用的 Agent 能力包
  2. Workspace — Agent 的工作目录管理
  3. Tracing — OpenTelemetry 分布式追踪

运行方式:
    .venv\Scripts\python.exe learn\09_skill_workspace_tracing.py
"""

import os
import sys
import asyncio
import time
import json
from pathlib import Path
from typing import Any

# 修复 Windows GBK 编码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 从 .env 文件加载 API Key
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

from agentscope.agent import Agent
from agentscope.agent._config import ReActConfig
from agentscope.model import OpenAIChatModel
from agentscope.credential import OpenAICredential
from agentscope.tool import ToolBase, Toolkit, ToolChunk
from agentscope.permission import (
    PermissionDecision,
    PermissionBehavior,
    PermissionContext,
)
from agentscope.message import TextBlock, UserMsg
from agentscope.skill import Skill, LocalSkillLoader
from agentscope.workspace import LocalWorkspace
from agentscope.middleware import TracingMiddleware


# ============================================================
# 第一部分: Skill — 可复用的 Agent 能力包
# ============================================================

def create_skill_directory():
    """
    创建 Skill 目录结构

    Skill 是一个包含 markdown 说明文件的目录，
    Agent 可以"学习"这些技能来扩展自己的能力。
    """
    skill_dir = Path(__file__).parent / "skills" / "code-reviewer"
    skill_dir.mkdir(parents=True, exist_ok=True)

    # 创建 Skill 说明文件
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("""---
name: Code Reviewer
description: 专业代码审查技能，帮助审查 Python 代码的质量、安全性和性能
---

# Code Reviewer Skill

你是一个专业的代码审查员。当需要审查代码时，请遵循以下步骤:

## 审查流程

1. **代码风格检查**
   - 命名规范（变量、函数、类）
   - 代码格式（缩进、空行）
   - 注释完整性

2. **功能正确性**
   - 逻辑是否正确
   - 边界条件处理
   - 错误处理机制

3. **性能优化**
   - 时间复杂度分析
   - 空间复杂度分析
   - 潜在性能瓶颈

4. **安全性检查**
   - SQL 注入风险
   - XSS 漏洞
   - 敏感信息泄露

## 输出格式

请按以下格式输出审查结果:

```
## 代码审查报告

### 代码风格
- [ ] 命名规范
- [ ] 代码格式
- [ ] 注释完整性

### 功能正确性
- [ ] 逻辑正确
- [ ] 边界条件
- [ ] 错误处理

### 性能优化
- [ ] 时间复杂度
- [ ] 空间复杂度

### 安全性
- [ ] SQL 注入
- [ ] XSS 漏洞
- [ ] 敏感信息

### 总结
[总体评价]
```
""", encoding="utf-8")

    # 创建示例代码文件
    example_code = skill_dir / "example.py"
    example_code.write_text("""# 示例代码 - 待审查

def calculate_average(numbers):
    if len(numbers) == 0:
        return 0
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)


def get_user_data(user_id):
    # TODO: 从数据库获取用户数据
    query = "SELECT * FROM users WHERE id = " + user_id
    return query


class UserManager:
    def __init__(self):
        self.users = []

    def add_user(self, user):
        self.users.append(user)

    def get_user(self, index):
        return self.users[index]
""", encoding="utf-8")

    print(f"[Skill] 已创建 Skill 目录: {skill_dir}")
    print(f"  - SKILL.md: Skill 说明文件")
    print(f"  - example.py: 示例代码")

    return skill_dir


async def demo_skill():
    """演示 Skill 的加载和使用"""
    print("\n" + "=" * 60)
    print("[第一部分] Skill — 可复用的 Agent 能力包")
    print("=" * 60)

    # 1. 创建 Skill 目录
    skill_dir = create_skill_directory()

    # 2. 使用 LocalSkillLoader 加载 Skill
    # scan_subdir=True 扫描子目录中的 SKILL.md
    loader = LocalSkillLoader(skill_dir.parent, scan_subdir=True)

    # 列出所有可用 Skills
    skills = await loader.list_skills()
    print(f"\n[Skill] 可用 Skills: {len(skills)} 个")
    for skill in skills:
        print(f"  - {skill.name}: {skill.description}")
        print(f"    目录: {skill.dir}")

    # 3. 创建带有 Skill 的 Agent
    # Skill 的内容会被注入到 Agent 的 system_prompt 中
    if skills:
        skill = skills[0]
        print(f"\n[Skill] 加载 Skill: {skill.name}")
        print(f"  Skill 内容预览:\n{skill.markdown[:200]}...")

        # 创建 Agent，将 Skill 内容注入 system_prompt
        agent = Agent(
            name="Code-Reviewer",
            system_prompt=(
                "你是一个专业的代码审查员。\n\n"
                f"## 技能说明\n{skill.markdown}\n\n"
                "请按照上述技能要求审查代码。"
            ),
            model=OpenAIChatModel(
                credential=OpenAICredential(
                    api_key=os.environ["OPENAI_API_KEY"],
                    base_url=os.environ.get("OPENAI_BASE_URL"),
                ),
                model="LongCat-2.0-Preview",
                stream=True,
                context_size=128000,
            ),
            toolkit=Toolkit(tools=[]),
            react_config=ReActConfig(max_iters=5),
        )

        # 使用 Agent 审查代码
        example_code = (Path(skill.dir) / "example.py").read_text(encoding="utf-8")
        print(f"\n[Skill] 使用 Agent 审查代码...")
        result = await agent.reply(
            UserMsg(name="user", content=f"请审查以下代码:\n\n```python\n{example_code}\n```")
        )
        print(f"\n[审查结果]\n{result.content}")

    return skills


# ============================================================
# 第二部分: Workspace — Agent 的工作目录管理
# ============================================================

async def demo_workspace():
    """演示 Workspace 的使用"""
    print("\n" + "=" * 60)
    print("[第二部分] Workspace — Agent 的工作目录管理")
    print("=" * 60)

    # 1. 创建本地 Workspace
    workspace_dir = Path(__file__).parent / "workspace"
    workspace_dir.mkdir(exist_ok=True)

    workspace = LocalWorkspace(workdir=str(workspace_dir))

    print(f"\n[Workspace] 工作目录: {workspace_dir}")

    # 2. 在 Workspace 中创建文件
    test_file = workspace_dir / "notes.txt"
    test_file.write_text(
        "这是 Agent 的工作笔记。\n"
        "今天学习了 AgentScope 的 Skill、Workspace 和 Tracing 功能。\n",
        encoding="utf-8"
    )
    print(f"\n[Workspace] 创建文件: {test_file}")

    # 3. 列出 Workspace 中的技能
    skills = await workspace.list_skills()
    print(f"\n[Workspace] 已加载技能: {len(skills)} 个")
    for skill in skills:
        print(f"  - {skill.name}: {skill.description}")

    # 4. 列出 Workspace 中的 MCP 客户端
    mcps = await workspace.list_mcps()
    print(f"\n[Workspace] 已加载 MCP: {len(mcps)} 个")
    for mcp in mcps:
        print(f"  - {mcp.name}")

    # 5. 创建带有文件操作工具的 Agent
    class FileReadTool(ToolBase):
        """文件读取工具"""
        name = "read_file"
        description = "读取工作目录中的文件。参数: filename-文件名"
        input_schema = {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "文件名"}
            },
            "required": ["filename"],
        }
        is_concurrency_safe = True
        is_read_only = True

        async def check_permissions(self, tool_input, context):
            return PermissionDecision(behavior=PermissionBehavior.ALLOW, message="允许")

        async def __call__(self, filename: str, **kwargs) -> ToolChunk:
            try:
                filepath = Path(workspace.workdir) / filename
                content = filepath.read_text(encoding="utf-8")
                return ToolChunk(content=[TextBlock(text=f"📄 {filename}:\n{content}")])
            except Exception as e:
                return ToolChunk(content=[TextBlock(text=f"读取失败: {e}")])

    class FileWriteTool(ToolBase):
        """文件写入工具"""
        name = "write_file"
        description = "写入文件到工作目录。参数: filename-文件名, content-内容"
        input_schema = {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "文件名"},
                "content": {"type": "string", "description": "文件内容"},
            },
            "required": ["filename", "content"],
        }
        is_concurrency_safe = False
        is_read_only = False

        async def check_permissions(self, tool_input, context):
            return PermissionDecision(behavior=PermissionBehavior.ALLOW, message="允许")

        async def __call__(self, filename: str, content: str, **kwargs) -> ToolChunk:
            try:
                filepath = Path(workspace.workdir) / filename
                filepath.write_text(content, encoding="utf-8")
                return ToolChunk(content=[TextBlock(text=f"已写入文件: {filename}")])
            except Exception as e:
                return ToolChunk(content=[TextBlock(text=f"写入失败: {e}")])

    agent = Agent(
        name="Workspace-Agent",
        system_prompt=(
            "你是一个文件管理助手。你可以:\n"
            "- read_file: 读取工作目录中的文件\n"
            "- write_file: 写入文件到工作目录\n"
            "用中文回答。"
        ),
        model=OpenAIChatModel(
            credential=OpenAICredential(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ.get("OPENAI_BASE_URL"),
            ),
            model="LongCat-2.0-Preview",
            stream=False,
            context_size=128000,
        ),
        toolkit=Toolkit(tools=[FileReadTool(), FileWriteTool()]),
        react_config=ReActConfig(max_iters=5),
    )

    # 6. 让 Agent 操作文件
    print(f"\n[Workspace] Agent 读取文件...")
    result = await agent.reply(
        UserMsg(name="user", content="请读取 notes.txt 文件的内容")
    )
    print(f"[Agent] {result.content}")

    print(f"\n[Workspace] Agent 写入文件...")
    result = await agent.reply(
        UserMsg(name="user", content="请在 workspace 中创建 todo.txt，内容: 1. 学习 Skill\n2. 学习 Workspace\n3. 学习 Tracing")
    )
    print(f"[Agent] {result.content}")

    # 7. 验证文件已创建
    workspace_path = Path(workspace.workdir)
    files = [f.name for f in workspace_path.iterdir() if f.is_file()]
    print(f"\n[Workspace] 更新后的文件列表:")
    for f in files:
        print(f"  - {f}")

    return workspace


# ============================================================
# 第三部分: Tracing — OpenTelemetry 分布式追踪
# ============================================================

async def demo_tracing():
    """演示 Tracing 中间件的使用"""
    print("\n" + "=" * 60)
    print("[第三部分] Tracing — OpenTelemetry 分布式追踪")
    print("=" * 60)

    # 1. 创建带 Tracing 的 Agent
    # TracingMiddleware 会自动记录:
    # - 每次推理的耗时
    # - 工具调用的输入输出
    # - Token 使用情况
    # - 错误信息

    tracing_mw = TracingMiddleware()

    # 创建一个简单的搜索工具
    class SearchTool(ToolBase):
        name = "search"
        description = "搜索知识库。"
        input_schema = {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "搜索关键词"}},
            "required": ["query"],
        }
        is_concurrency_safe = True
        is_read_only = True

        async def check_permissions(self, tool_input, context):
            return PermissionDecision(behavior=PermissionBehavior.ALLOW, message="允许")

        async def __call__(self, query: str, **kwargs) -> ToolChunk:
            knowledge = {
                "agentscope": "AgentScope 是阿里巴巴开源的多智能体框架 2.0。",
                "tracing": "Tracing 是 OpenTelemetry 提供的分布式追踪功能。",
                "skill": "Skill 是 AgentScope 的可复用能力包机制。",
                "workspace": "Workspace 是 Agent 的工作目录管理功能。",
            }
            for key, value in knowledge.items():
                if key in query.lower():
                    return ToolChunk(content=[TextBlock(text=value)])
            return ToolChunk(content=[TextBlock(text=f"未找到 '{query}' 的信息")])

    agent = Agent(
        name="Tracing-Demo",
        system_prompt=(
            "你是一个助手。使用 search 工具搜索信息。\n"
            "用中文回答，回答要简洁。"
        ),
        model=OpenAIChatModel(
            credential=OpenAICredential(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ.get("OPENAI_BASE_URL"),
            ),
            model="LongCat-2.0-Preview",
            stream=False,
            context_size=128000,
        ),
        toolkit=Toolkit(tools=[SearchTool()]),
        react_config=ReActConfig(max_iters=5),
        # 挂载 Tracing 中间件
        middlewares=[tracing_mw],
    )

    # 2. 执行一些操作，Tracing 会自动记录
    print("\n[Tracing] 执行 Agent 操作...")

    questions = [
        "什么是 AgentScope?",
        "什么是 Tracing?",
    ]

    for q in questions:
        print(f"\n  用户: {q}")
        start = time.time()
        result = await agent.reply(UserMsg(name="user", content=q))
        elapsed = time.time() - start
        print(f"  Agent: {result.content}")
        print(f"  耗时: {elapsed:.2f}s")

    # 3. 打印 Agent 状态（包含追踪信息）
    print(f"\n[Tracing] Agent 状态:")
    print(f"  名称: {agent.name}")
    print(f"  上下文消息数: {len(agent.state.context)}")
    print(f"  当前回复 ID: {agent.state.reply_id}")
    print(f"  会话 ID: {agent.state.session_id[:16]}...")

    return agent


# ============================================================
# 主程序
# ============================================================
async def main():
    print("AgentScope 高级功能演示")
    print("Skill + Workspace + Tracing")
    print("=" * 60)

    # 第一部分: Skill
    await demo_skill()

    # 第二部分: Workspace
    await demo_workspace()

    # 第三部分: Tracing
    await demo_tracing()

    # 总结
    print("\n" + "=" * 60)
    print("[总结]")
    print("=" * 60)
    print("""
三大高级功能:

1. Skill (技能包)
   - 将 Agent 的能力封装为可复用的 markdown 文件
   - 通过 LocalSkillLoader 加载
   - 内容注入到 system_prompt 中
   - 用途: 代码审查、数据分析、报告生成等专业技能

2. Workspace (工作目录)
   - 为 Agent 提供文件系统访问能力
   - LocalWorkspace: 本地文件系统
   - DockerWorkspace: Docker 容器内文件系统
   - E2BWorkspace: 云端沙箱文件系统
   - 用途: 文件读写、代码执行、数据处理

3. Tracing (分布式追踪)
   - 基于 OpenTelemetry 标准
   - 自动记录推理耗时、工具调用、Token 使用
   - 支持导出到 Jaeger、Zipkin 等追踪系统
   - 用途: 性能监控、问题排查、成本分析
    """)


if __name__ == "__main__":
    asyncio.run(main())
