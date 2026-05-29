# -*- coding: utf-8 -*-
"""
综合实战: 智能客服系统
======================
整合 AgentScope 所有核心功能:
  1. ReAct Agent — 推理+行动循环
  2. MCP 工具集成 — 外部工具协议
  3. A2A 多智能体协作 — 多 Agent 配合
  4. Skill 技能包 — 可复用能力
  5. Tracing 链路追踪 — 性能监控
  6. Workspace 工作空间 — 文件管理

场景: 用户咨询技术问题
  用户输入 → 意图识别 Agent → 分配给专业 Agent → 搜索知识库 → 生成回答

运行方式:
    .venv\Scripts\python.exe learn\10_full_project.py
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
from agentscope.permission import PermissionDecision, PermissionBehavior, PermissionContext
from agentscope.message import TextBlock, UserMsg
from agentscope.middleware import MiddlewareBase
from agentscope.event import (
    AgentEvent,
    ModelCallStartEvent,
    ModelCallEndEvent,
    ToolCallStartEvent,
    ToolResultEndEvent,
    ReplyEndEvent,
)
from agentscope.skill import LocalSkillLoader
from agentscope.workspace import LocalWorkspace


# ============================================================
# 第一部分: Tracing 中间件（复用之前写的）
# ============================================================
class SimpleTracingMiddleware(MiddlewareBase):
    """简易追踪中间件"""

    def __init__(self, agent_name=""):
        self.logs = []
        self._reply_start = None
        self._agent_name = agent_name

    def _log(self, msg: str):
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] [{self._agent_name}] {msg}"
        self.logs.append(entry)

    async def on_reply(self, agent, input_kwargs, next_handler):
        self._log("=== 回复开始 ===")
        self._reply_start = time.time()

        async for event in next_handler(**input_kwargs):
            event_type = type(event).__name__

            if isinstance(event, ModelCallStartEvent):
                self._log(f"[推理] 模型: {event.model_name}")

            elif isinstance(event, ModelCallEndEvent):
                self._log(
                    f"[推理] 输入tokens: {event.input_tokens}, "
                    f"输出tokens: {event.output_tokens}"
                )

            elif isinstance(event, ToolCallStartEvent):
                self._log(f"[工具] 调用: {event.tool_call_name}")

            elif isinstance(event, ToolResultEndEvent):
                self._log(f"[工具] 返回: {event.state}")

            elif isinstance(event, ReplyEndEvent):
                elapsed = time.time() - self._reply_start
                self._log(f"[完成] 耗时: {elapsed:.2f}s")

            yield event

        elapsed = time.time() - self._reply_start
        self._log(f"=== 回复完成 | 总耗时: {elapsed:.2f}s ===\n")


# ============================================================
# 第二部分: MCP 工具（模拟外部服务）
# ============================================================
class MCPKnowledgeBaseTool(ToolBase):
    """模拟 MCP 知识库搜索工具"""

    name = "mcp_search_kb"
    description = "搜索技术知识库。参数: query-搜索关键词"
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "搜索关键词"}},
        "required": ["query"],
    }
    is_concurrency_safe = True
    is_read_only = True
    is_mcp = True
    mcp_name = "knowledge-base"

    _knowledge = {
        "react": "ReAct = Reasoning + Acting。Agent 交替进行思考和行动，直到完成任务。核心思想: 边想边做。",
        "rag": "RAG = Retrieval-Augmented Generation。先检索外部知识，再交给 LLM 生成回答。优点: 知识可更新，减少幻觉。",
        "mcp": "MCP = Model Context Protocol。标准化 LLM 与外部工具的通信协议。优点: 工具可复用，跨模型兼容。",
        "a2a": "A2A = Agent-to-Agent。Google 提出的 Agent 间通信协议。用途: 多智能体协作。",
        "agentscope": "AgentScope 是阿里巴巴开源的多智能体框架 2.0。支持 ReAct、MCP、A2A、Skill、Tracing 等。",
        "python": "Python 是一种高级编程语言，广泛用于 AI、数据科学、Web 开发。特点: 简洁易读，生态丰富。",
        "asyncio": "Python asyncio 是异步编程库，用于编写并发代码。核心: 事件循环 + async/await。",
        "fastapi": "FastAPI 是现代 Python Web 框架，用于构建 API。特点: 高性能，自动生成文档。",
        "docker": "Docker 是容器化平台，用于打包和部署应用。核心: 镜像 + 容器。",
        "kubernetes": "Kubernetes (K8s) 是容器编排平台，用于自动化部署、扩展和管理容器。",
    }

    async def check_permissions(self, tool_input, context):
        return PermissionDecision(behavior=PermissionBehavior.ALLOW, message="允许")

    async def __call__(self, query: str, **kwargs) -> ToolChunk:
        results = []
        for key, value in self._knowledge.items():
            if key in query.lower():
                results.append(f"[{key.upper()}] {value}")
        if results:
            return ToolChunk(content=[TextBlock(text="\n\n".join(results))])
        return ToolChunk(content=[TextBlock(text=f"知识库中未找到 '{query}' 的相关信息")])


class MCPCodeExecutorTool(ToolBase):
    """模拟 MCP 代码执行工具"""

    name = "mcp_execute_code"
    description = "执行 Python 代码并返回结果。参数: code-要执行的代码"
    input_schema = {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python 代码"}},
        "required": ["code"],
    }
    is_concurrency_safe = False
    is_read_only = False
    is_mcp = True
    mcp_name = "code-executor"

    async def check_permissions(self, tool_input, context):
        return PermissionDecision(behavior=PermissionBehavior.ALLOW, message="允许")

    async def __call__(self, code: str, **kwargs) -> ToolChunk:
        try:
            # 安全执行: 只用基本运算
            allowed = {"abs": abs, "max": max, "min": min, "round": round, "sum": sum, "len": len}
            result = eval(code, {"__builtins__": {}}, allowed)
            return ToolChunk(content=[TextBlock(text=f"执行结果: {result}")])
        except Exception as e:
            return ToolChunk(content=[TextBlock(text=f"执行错误: {e}")])


# ============================================================
# 第三部分: Skill 技能包
# ============================================================
def setup_skills():
    """创建 Skill 目录"""
    skill_dir = Path(__file__).parent / "skills" / "tech-support"
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("""---
name: Tech Support
description: 技术支持技能，帮助回答技术问题
---

# 技术支持技能

你是一个专业的技术支持工程师。请遵循以下原则:

## 回答原则
1. 准确 — 确保技术信息正确
2. 简洁 — 用简单的语言解释复杂概念
3. 实用 — 提供可操作的建议
4. 完整 — 覆盖问题的所有方面

## 回答结构
- 概念解释: 用 1-2 句话解释核心概念
- 工作原理: 简要说明工作原理
- 使用场景: 列举典型应用场景
- 示例: 提供简单示例（如果适用）
- 总结: 一句话总结

## 语言
- 用中文回答
- 专业术语保留英文原文
""", encoding="utf-8")

    print(f"[Skill] 已创建: {skill_dir}")
    return skill_dir


# ============================================================
# 第四部分: 多智能体系统
# ============================================================
def create_agent(name: str, system_prompt: str, tools: list, tracer: SimpleTracingMiddleware) -> Agent:
    """创建带追踪的 Agent"""
    return Agent(
        name=name,
        system_prompt=system_prompt,
        model=OpenAIChatModel(
            credential=OpenAICredential(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ.get("OPENAI_BASE_URL"),
            ),
            model="LongCat-2.0-Preview",
            stream=False,
            context_size=128000,
        ),
        toolkit=Toolkit(tools=tools),
        react_config=ReActConfig(max_iters=10),
        middlewares=[tracer],
    )


async def main():
    print("=" * 70)
    print("  综合实战: 智能客服系统")
    print("  整合 ReAct + MCP + A2A + Skill + Tracing + Workspace")
    print("=" * 70)

    # ========================================
    # 步骤 1: 初始化 Workspace
    # ========================================
    print("\n" + "─" * 50)
    print("[步骤 1] 初始化 Workspace")
    print("─" * 50)

    workspace_dir = Path(__file__).parent / "workspace_project"
    workspace_dir.mkdir(exist_ok=True)

    workspace = LocalWorkspace(workdir=str(workspace_dir))
    await workspace.initialize()

    print(f"  Workspace ID: {workspace.workspace_id[:16]}...")
    print(f"  工作目录: {workspace_dir}")
    print(f"  状态: is_alive={workspace.is_alive}")

    # ========================================
    # 步骤 2: 加载 Skill
    # ========================================
    print("\n" + "─" * 50)
    print("[步骤 2] 加载 Skill")
    print("─" * 50)

    skill_dir = setup_skills()
    loader = LocalSkillLoader(str(skill_dir.parent), scan_subdir=True)
    skills = await loader.list_skills()

    print(f"  加载了 {len(skills)} 个 Skill:")
    for skill in skills:
        print(f"    - {skill.name}: {skill.description}")

    # ========================================
    # 步骤 3: 创建 MCP 工具
    # ========================================
    print("\n" + "─" * 50)
    print("[步骤 3] 创建 MCP 工具")
    print("─" * 50)

    mcp_tools = [
        MCPKnowledgeBaseTool(),
        MCPCodeExecutorTool(),
    ]
    print(f"  创建了 {len(mcp_tools)} 个 MCP 工具:")
    for tool in mcp_tools:
        print(f"    - {tool.name} (来自 {tool.mcp_name})")

    # ========================================
    # 步骤 4: 创建多智能体系统 (A2A)
    # ========================================
    print("\n" + "─" * 50)
    print("[步骤 4] 创建多智能体系统 (A2A)")
    print("─" * 50)

    # 意图识别 Agent
    intent_tracer = SimpleTracingMiddleware("意图识别")
    intent_agent = create_agent(
        name="意图识别",
        system_prompt=(
            "你是一个意图识别专家。你的任务是分析用户输入，判断用户需要哪类帮助。\n"
            "分类:\n"
            "- concept: 概念解释（如: 什么是 ReAct?）\n"
            "- comparison: 对比分析（如: ReAct 和 RAG 有什么区别?）\n"
            "- howto: 操作指南（如: 如何使用 FastAPI?）\n"
            "- code: 代码相关（如: 写一个 Python 异步函数）\n"
            "- other: 其他\n"
            "\n请只输出分类名称，不要输出其他内容。"
        ),
        tools=[],
        tracer=intent_tracer,
    )

    # 概念解释 Agent (带 Skill)
    concept_tracer = SimpleTracingMiddleware("概念专家")
    concept_agent = create_agent(
        name="概念专家",
        system_prompt=(
            "你是一个技术概念解释专家。\n\n"
            f"## 技能说明\n{skills[0].markdown if skills else ''}\n\n"
            "请使用 mcp_search_kb 工具搜索知识库，然后根据技能要求回答问题。\n"
            "用中文回答。"
        ),
        tools=mcp_tools,
        tracer=concept_tracer,
    )

    # 代码专家 Agent
    code_tracer = SimpleTracingMiddleware("代码专家")
    code_agent = create_agent(
        name="代码专家",
        system_prompt=(
            "你是一个 Python 代码专家。\n"
            "你可以:\n"
            "- 使用 mcp_search_kb 搜索知识库\n"
            "- 使用 mcp_execute_code 执行代码\n"
            "用中文回答，代码用 Python。"
        ),
        tools=mcp_tools,
        tracer=code_tracer,
    )

    # 综合回答 Agent (A2A 协调者)
    coordinator_tracer = SimpleTracingMiddleware("协调者")
    coordinator_agent = create_agent(
        name="协调者",
        system_prompt=(
            "你是一个技术支持协调者。\n"
            "你可以将任务分配给专业 Agent:\n"
            "- 概念专家: 负责概念解释类问题\n"
            "- 代码专家: 负责代码相关的问题\n"
            "\n根据用户问题，选择合适的 Agent 回答。\n"
            "用中文回答。"
        ),
        tools=mcp_tools,
        tracer=coordinator_tracer,
    )

    agents = {
        "意图识别": intent_agent,
        "概念专家": concept_agent,
        "代码专家": code_agent,
        "协调者": coordinator_agent,
    }

    print(f"  创建了 {len(agents)} 个 Agent:")
    for name in agents:
        print(f"    - {name}")

    # ========================================
    # 步骤 5: 运行智能客服系统
    # ========================================
    print("\n" + "─" * 50)
    print("[步骤 5] 运行智能客服系统")
    print("─" * 50)

    test_questions = [
        "什么是 ReAct? 请详细解释。",
        "ReAct 和 RAG 有什么区别?",
        "计算 2 的 100 次方是多少?",
    ]

    all_logs = []

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'=' * 60}")
        print(f"[用户问题 {i}] {question}")
        print("=" * 60)

        # 步骤 5.1: 意图识别
        print(f"\n  [1/3] 意图识别中...")
        intent_result = await intent_agent.reply(
            UserMsg(name="user", content=question)
        )
        intent = intent_result.content.strip().lower() if isinstance(intent_result.content, str) else str(intent_result.content).strip().lower()
        print(f"  识别结果: {intent}")

        # 步骤 5.2: 分配给专业 Agent (A2A)
        print(f"\n  [2/3] 分配给专业 Agent...")

        if "code" in intent or "计算" in question or "次方" in question:
            target_agent = code_agent
            agent_type = "代码专家"
        else:
            target_agent = concept_agent
            agent_type = "概念专家"

        print(f"  选择: {agent_type}")

        # A2A 消息传递: 意图识别结果 → 专业 Agent
        agent_msg = UserMsg(
            name="意图识别",
            content=f"用户问题: {question}\n意图分类: {intent}\n请回答用户的问题。",
        )

        # 步骤 5.3: 专业 Agent 回答
        print(f"\n  [3/3] {agent_type} 回答中...")
        answer = await target_agent.reply(agent_msg)
        print(f"\n  [回答]\n  {answer.content}")

        # 收集追踪日志
        for tracer in [intent_tracer, concept_tracer, code_tracer, coordinator_tracer]:
            all_logs.extend(tracer.logs)

    # ========================================
    # 步骤 6: 打印追踪摘要
    # ========================================
    print("\n" + "=" * 70)
    print("[追踪摘要]")
    print("=" * 70)

    # 统计每个 Agent 的耗时
    agent_stats = {}
    for log in all_logs:
        if "回复完成" in log and "耗时:" in log:
            parts = log.split("]")
            agent_name = parts[1].strip() if len(parts) > 1 else "unknown"
            time_str = log.split("耗时:")[-1].replace("s ===", "").strip()
            try:
                elapsed = float(time_str)
                if agent_name not in agent_stats:
                    agent_stats[agent_name] = {"count": 0, "total_time": 0}
                agent_stats[agent_name]["count"] += 1
                agent_stats[agent_name]["total_time"] += elapsed
            except ValueError:
                pass

    print("\n  Agent 耗时统计:")
    for agent_name, stats in agent_stats.items():
        avg_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
        print(f"    {agent_name}: {stats['count']} 次调用, 平均 {avg_time:.2f}s")

    # 统计 Token 使用
    total_input_tokens = 0
    total_output_tokens = 0
    for log in all_logs:
        if "输入tokens:" in log:
            try:
                input_tokens = int(log.split("输入tokens:")[-1].split(",")[0].strip())
                output_tokens = int(log.split("输出tokens:")[-1].strip())
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
            except (ValueError, IndexError):
                pass

    print(f"\n  Token 使用统计:")
    print(f"    输入 tokens: {total_input_tokens}")
    print(f"    输出 tokens: {total_output_tokens}")
    print(f"    总计 tokens: {total_input_tokens + total_output_tokens}")

    # ========================================
    # 步骤 7: 清理
    # ========================================
    await workspace.close()
    print(f"\n  Workspace 已关闭")

    # ========================================
    # 总结
    # ========================================
    print("\n" + "=" * 70)
    print("[项目总结]")
    print("=" * 70)
    print("""
  本项目整合了 AgentScope 的所有核心功能:

  1. ReAct Agent
     - 每个 Agent 都使用 ReAct 循环 (推理+行动)
     - 自动调用工具获取信息并生成回答

  2. MCP 工具集成
     - MCPKnowledgeBaseTool: 模拟知识库搜索
     - MCPCodeExecutorTool: 模拟代码执行
     - 工具标记 is_mcp=True, mcp_name="xxx"

  3. A2A 多智能体协作
     - 意图识别 Agent → 专业 Agent 的消息传递
     - 根据意图分类选择合适的 Agent 回答
     - UserMsg.name 标识发送者

  4. Skill 技能包
     - 从 skills/ 目录加载 Tech Support 技能
     - 技能内容注入到 Agent 的 system_prompt

  5. Tracing 链路追踪
     - 记录每次推理的耗时和 Token 使用
     - 追踪工具调用的输入输出
     - 统计每个 Agent 的性能指标

  6. Workspace 工作空间
     - 持久化的工作环境
     - 自动加载技能和 MCP 配置
     - 会话数据管理

  架构图:
     ┌─────────────────────────────────────────────────────────┐
     │                    用户输入                              │
     └───────────────────────┬─────────────────────────────────┘
                             ▼
     ┌─────────────────────────────────────────────────────────┐
     │              意图识别 Agent (ReAct)                      │
     │  分析用户问题 → 分类: concept/code/comparison/other      │
     └───────────────────────┬─────────────────────────────────┘
                             ▼ (A2A 消息传递)
     ┌─────────────────────────────────────────────────────────┐
     │         专业 Agent (概念专家 / 代码专家)                  │
     │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
     │  │  Skill 技能  │  │  MCP 工具   │  │  Tracing    │     │
     │  │  注入指导    │  │  搜索/执行   │  │  性能监控   │     │
     │  └─────────────┘  └─────────────┘  └─────────────┘     │
     └───────────────────────┬─────────────────────────────────┘
                             ▼
     ┌─────────────────────────────────────────────────────────┐
     │                   生成回答                               │
     └─────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    asyncio.run(main())
