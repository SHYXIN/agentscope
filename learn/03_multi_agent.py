# -*- coding: utf-8 -*-
"""
实战 3: 多智能体协作
====================
场景: 两个 Agent 协作完成一个任务
- 研究员 (Researcher): 负责搜索和收集信息
- 写手 (Writer): 负责整理和输出最终答案

协作流程:
  用户问题 -> 研究员搜索信息 -> 研究员把结果"告诉"写手 -> 写手整理输出

运行方式:
    D:\code_project\github-proj\agentscope\.venv\Scripts\python.exe learn\03_multi_agent.py
"""

import os
import sys
import asyncio
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
from agentscope.message import TextBlock, UserMsg, Msg


# ============================================================
# 工具: 知识搜索 (研究员专用)
# ============================================================
class ResearchTool(ToolBase):
    """搜索知识库获取研究资料"""

    name = "search_knowledge"
    description = (
        "搜索知识库获取研究资料。当需要查询某个概念、"
        "技术细节或背景知识时使用。"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词",
            }
        },
        "required": ["query"],
    }
    is_concurrency_safe = True
    is_read_only = True

    async def check_permissions(self, tool_input, context):
        return PermissionDecision(
            behavior=PermissionBehavior.ALLOW,
            message="允许执行",
        )

    async def __call__(self, query: str, **kwargs) -> ToolChunk:
        knowledge = {
            "react": (
                "ReAct (Reasoning + Acting) 是一种 Agent 范式。\n"
                "核心思想: 交替进行推理和行动。\n"
                "优点: 可以调用外部工具获取实时信息。\n"
                "应用: 问答系统、代码生成、数据分析。"
            ),
            "rag": (
                "RAG (Retrieval-Augmented Generation) 检索增强生成。\n"
                "核心思想: 先检索相关文档，再交给 LLM 生成回答。\n"
                "优点: 知识可更新，减少幻觉。\n"
                "应用: 知识库问答、文档理解。"
            ),
            "mcp": (
                "MCP (Model Context Protocol) 模型上下文协议。\n"
                "核心思想: 标准化 LLM 与外部工具的通信协议。\n"
                "优点: 工具可复用，跨模型兼容。\n"
                "应用: 工具集成、资源访问。"
            ),
            "agent": (
                "Agent (智能体) 是能感知环境、自主决策的系统。\n"
                "核心能力: 推理、规划、工具使用、记忆。\n"
                "类型: 单 Agent、多 Agent、ReAct Agent。\n"
                "应用: 自动化任务、智能助手、代码生成。"
            ),
        }
        for key, value in knowledge.items():
            if key in query.lower():
                return ToolChunk(content=[TextBlock(text=value)])
        return ToolChunk(content=[
            TextBlock(text=f"未找到关于 '{query}' 的详细资料")
        ])


# ============================================================
# 工具: 计算器 (两个 Agent 都能用)
# ============================================================
class CalculatorTool(ToolBase):
    name = "calculator"
    description = "执行数学计算。"
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式",
            }
        },
        "required": ["expression"],
    }
    is_concurrency_safe = True
    is_read_only = True

    async def check_permissions(self, tool_input, context):
        return PermissionDecision(
            behavior=PermissionBehavior.ALLOW,
            message="允许执行",
        )

    async def __call__(self, expression: str, **kwargs) -> ToolChunk:
        try:
            allowed = {"abs": abs, "max": max, "min": min, "round": round}
            result = eval(expression, {"__builtins__": {}}, allowed)
            return ToolChunk(content=[TextBlock(text=f"结果: {result}")])
        except Exception as e:
            return ToolChunk(content=[TextBlock(text=f"错误: {e}")])


# ============================================================
# 辅助函数: 流式输出 Agent 回复
# ============================================================
async def stream_reply(agent: Agent, user_msg: UserMsg, label: str = ""):
    """流式输出 Agent 的回复，返回完整回复文本"""
    if label:
        print(f"\n{'='*60}")
        print(f"[{label}]")
        print(f"{'='*60}")

    full_text = ""
    tool_args_buffer = ""
    tool_result_buffer = ""

    async for event in agent.reply_stream(user_msg):
        event_type = type(event).__name__

        if event_type == "TextBlockDeltaEvent":
            print(event.delta, end="", flush=True)
            full_text += event.delta

        elif event_type == "ThinkingBlockDeltaEvent":
            print(f"\n  [思考]: {event.delta}", end="", flush=True)

        elif event_type == "ToolCallStartEvent":
            print(f"\n  [调用工具] {event.tool_call_name}")
            tool_args_buffer = ""

        elif event_type == "ToolCallDeltaEvent":
            tool_args_buffer += event.delta

        elif event_type == "ToolCallEndEvent":
            print(f"  [工具参数] {tool_args_buffer}")
            tool_result_buffer = ""

        elif event_type == "ToolResultTextDeltaEvent":
            tool_result_buffer += event.delta

        elif event_type == "ToolResultEndEvent":
            print(f"  [工具返回] {tool_result_buffer}")

        elif event_type == "ReplyEndEvent":
            print("\n")

    return full_text


# ============================================================
# 主程序: 多智能体协作
# ============================================================
async def main():
    # 创建共享模型
    credential = OpenAICredential(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    model = OpenAIChatModel(
        credential=credential,
        model="LongCat-2.0-Preview",
        stream=True,
        context_size=128000,
    )

    # 1. 创建研究员 Agent (有搜索工具)
    researcher = Agent(
        name="研究员",
        system_prompt=(
            "你是一个专业的研究员。你的职责是:\n"
            "1. 使用 search_knowledge 工具搜索和收集信息\n"
            "2. 整理成结构化的研究报告\n"
            "3. 将报告传递给写手\n"
            "请用中文回答。"
        ),
        model=model,
        toolkit=Toolkit(tools=[ResearchTool(), CalculatorTool()]),
        react_config=ReActConfig(max_iters=10),
    )

    # 2. 创建写手 Agent (没有搜索工具，依赖研究员的信息)
    writer = Agent(
        name="写手",
        system_prompt=(
            "你是一个专业的技术写手。你的职责是:\n"
            "1. 接收研究员提供的信息\n"
            "2. 整理成通俗易懂的回答\n"
            "3. 确保回答结构清晰、逻辑严谨\n"
            "请用中文回答。"
        ),
        model=model,
        toolkit=Toolkit(tools=[CalculatorTool()]),
        react_config=ReActConfig(max_iters=10),
    )

    # 3. 测试: 多智能体协作
    question = "请解释什么是 ReAct 和 RAG，以及它们的关系。"

    print("=" * 60)
    print(f"[用户问题] {question}")
    print("=" * 60)

    # 阶段 1: 研究员搜索信息
    research_msg = UserMsg(
        name="user",
        content=f"请搜索以下主题的资料，整理成结构化报告:\n{question}"
    )
    research_report = await stream_reply(
        researcher, research_msg, label="阶段 1: 研究员搜索信息"
    )

    # 阶段 2: 研究员把结果"告诉"写手
    # 使用 observe 方法让写手接收研究员的报告
    handoff_msg = UserMsg(
        name="研究员",
        content=(
            f"这是你的研究报告，请基于这些信息回答用户的问题:\n\n"
            f"{research_report}\n\n"
            f"用户原始问题: {question}"
        )
    )

    # 阶段 3: 写手整理输出
    final_answer = await stream_reply(
        writer, handoff_msg, label="阶段 2: 写手整理输出"
    )

    # 4. 再测试一轮: 写手可以追问研究员
    print("\n" + "=" * 60)
    print("[第二轮: 写手追问研究员]")
    print("=" * 60)

    follow_up = UserMsg(
        name="user",
        content="MCP 协议有什么优势？请详细说明。"
    )
    research_report_2 = await stream_reply(
        researcher, follow_up, label="研究员补充搜索"
    )

    handoff_msg_2 = UserMsg(
        name="研究员",
        content=(
            f"补充资料:\n{research_report_2}\n\n"
            f"请结合之前的资料和这些新信息，回答用户的问题。"
        )
    )
    final_answer_2 = await stream_reply(
        writer, handoff_msg_2, label="写手最终回答"
    )


if __name__ == "__main__":
    asyncio.run(main())
