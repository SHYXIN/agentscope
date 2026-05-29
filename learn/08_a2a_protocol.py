# -*- coding: utf-8 -*-
"""
实战 8: A2A (Agent-to-Agent) 协议
===================================
A2A 是 Google 提出的 Agent 间通信协议，让不同 Agent 可以互相调用。

AgentScope 2.0 通过以下方式支持 A2A:
  1. Pipeline — 串行/并行编排多个 Agent
  2. AGUI 协议 — 将 Agent 事件流转换为标准协议格式
  3. 消息传递 — Agent 间通过消息通信

本实战演示:
  1. Pipeline 串行编排 (Agent A → Agent B → Agent C)
  2. Pipeline 并行编排 (多个 Agent 同时工作)
  3. Agent 间消息传递

运行方式:
    .venv\Scripts\python.exe learn\08_a2a_protocol.py
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
from agentscope.message import TextBlock, UserMsg


# ============================================================
# 工具定义
# ============================================================
class SearchTool(ToolBase):
    """搜索工具"""
    name = "search"
    description = "搜索知识库获取信息。"
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
            "python": "Python 是一种高级编程语言，广泛用于 AI、数据科学。",
            "react": "ReAct = Reasoning + Acting，Agent 交替进行思考和行动。",
            "rag": "RAG = Retrieval-Augmented Generation，先检索再生成。",
            "mcp": "MCP = Model Context Protocol，标准化 LLM 与工具的通信协议。",
            "a2a": "A2A = Agent-to-Agent，Google 提出的 Agent 间通信协议。",
            "agentscope": "AgentScope 是阿里巴巴开源的多智能体框架。",
        }
        for key, value in knowledge.items():
            if key in query.lower():
                return ToolChunk(content=[TextBlock(text=value)])
        return ToolChunk(content=[TextBlock(text=f"未找到 '{query}' 的信息")])


# ============================================================
# 辅助函数: 创建 Agent
# ============================================================
def create_agent(name: str, system_prompt: str) -> Agent:
    """创建 Agent"""
    return Agent(
        name=name,
        system_prompt=system_prompt,
        model=OpenAIChatModel(
            credential=OpenAICredential(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ.get("OPENAI_BASE_URL"),
            ),
            model="LongCat-2.0-Preview",
            stream=False,  # 非流式，方便 Pipeline 编排
            context_size=128000,
        ),
        toolkit=Toolkit(tools=[SearchTool()]),
        react_config=ReActConfig(max_iters=10),
    )


# ============================================================
# 方式 1: 串行 Pipeline (Agent A → Agent B → Agent C)
# ============================================================
async def serial_pipeline():
    """
    串行 Pipeline: 多个 Agent 按顺序处理同一个任务

    流程:
      用户输入 → Agent 1 (分析) → Agent 2 (翻译) → Agent 3 (校对) → 最终输出
    """
    print("=" * 60)
    print("[方式 1] 串行 Pipeline — 多 Agent 协作处理任务")
    print("=" * 60)

    # 创建 3 个专门化的 Agent
    analyzer = create_agent(
        name="分析员",
        system_prompt=(
            "你是一个信息分析员。你的职责是:\n"
            "1. 使用 search 工具搜索相关信息\n"
            "2. 提取关键信息，整理成结构化报告\n"
            "3. 将报告传递给下一个 Agent\n"
            "用中文回答。"
        ),
    )

    translator = create_agent(
        name="翻译员",
        system_prompt=(
            "你是一个翻译员。你的职责是:\n"
            "1. 接收分析员的报告\n"
            "2. 将报告翻译成英文\n"
            "3. 保持专业术语的准确性\n"
            "用中文回复确认。"
        ),
    )

    reviewer = create_agent(
        name="校对员",
        system_prompt=(
            "你是一个校对员。你的职责是:\n"
            "1. 接收翻译员的英文报告\n"
            "2. 检查翻译质量和准确性\n"
            "3. 输出最终校对后的报告\n"
            "用中文回复确认。"
        ),
    )

    # 用户输入
    user_input = "请解释什么是 ReAct 和 RAG，以及它们的关系。"
    print(f"\n[用户输入] {user_input}\n")

    # 步骤 1: 分析员搜索并整理信息
    print("─" * 40)
    print("[步骤 1] 分析员搜索信息...")
    print("─" * 40)
    msg1 = UserMsg(name="user", content=f"请搜索以下主题的资料，整理成结构化报告:\n{user_input}")
    result1 = await analyzer.reply(msg1)
    print(f"\n[分析员报告]\n{result1.content}\n")

    # 步骤 2: 翻译员翻译报告
    print("─" * 40)
    print("[步骤 2] 翻译员翻译报告...")
    print("─" * 40)
    msg2 = UserMsg(
        name="分析员",
        content=f"请将以下报告翻译成英文:\n\n{result1.content}",
    )
    result2 = await translator.reply(msg2)
    print(f"\n[翻译员输出]\n{result2.content}\n")

    # 步骤 3: 校对员校对
    print("─" * 40)
    print("[步骤 3] 校对员校对...")
    print("─" * 40)
    msg3 = UserMsg(
        name="翻译员",
        content=f"请校对以下翻译报告:\n\n{result2.content}",
    )
    result3 = await reviewer.reply(msg3)
    print(f"\n[校对员输出]\n{result3.content}\n")

    return result3.content


# ============================================================
# 方式 2: 并行 Pipeline (多个 Agent 同时工作)
# ============================================================
async def parallel_pipeline():
    """
    并行 Pipeline: 多个 Agent 同时处理不同子任务

    流程:
      用户输入 → 拆分为多个子任务 → 多个 Agent 并行处理 → 合并结果
    """
    print("\n" + "=" * 60)
    print("[方式 2] 并行 Pipeline — 多 Agent 同时工作")
    print("=" * 60)

    # 创建 3 个专家 Agent
    expert_python = create_agent(
        name="Python专家",
        system_prompt="你是 Python 专家。用中文简洁回答，不超过 100 字。",
    )

    expert_react = create_agent(
        name="ReAct专家",
        system_prompt="你是 ReAct 专家。用中文简洁回答，不超过 100 字。",
    )

    expert_rag = create_agent(
        name="RAG专家",
        system_prompt="你是 RAG 专家。用中文简洁回答，不超过 100 字。",
    )

    # 用户输入
    print("\n[用户输入] 请分别解释 Python、ReAct 和 RAG\n")

    # 并行发送请求
    print("─" * 40)
    print("[并行处理] 3 个专家同时工作...")
    print("─" * 40)

    results = await asyncio.gather(
        expert_python.reply(UserMsg(name="user", content="什么是 Python?")),
        expert_react.reply(UserMsg(name="user", content="什么是 ReAct?")),
        expert_rag.reply(UserMsg(name="user", content="什么是 RAG?")),
    )

    # 合并结果
    combined = "\n".join(
        f"[{r.name}]\n{r.content}\n" for r in results
    )
    print(f"\n[合并结果]\n{combined}")

    return combined


# ============================================================
# 方式 3: Agent 间直接消息传递 (A2A 核心)
# ============================================================
async def agent_to_agent_messaging():
    """
    A2A 核心: Agent 之间直接传递消息

    流程:
      Agent A 完成任务 → 将结果作为消息发送给 Agent B
      → Agent B 基于 Agent A 的结果继续工作
    """
    print("\n" + "=" * 60)
    print("[方式 3] Agent 间直接消息传递 (A2A)")
    print("=" * 60)

    # 创建 Agent A (信息收集者)
    agent_a = create_agent(
        name="信息收集者",
        system_prompt=(
            "你是一个信息收集者。\n"
            "1. 使用 search 工具搜索信息\n"
            "2. 将收集到的信息整理成简洁的要点\n"
            "3. 用 list 格式输出，每条不超过 30 字\n"
            "用中文回答。"
        ),
    )

    # 创建 Agent B (报告撰写者)
    agent_b = create_agent(
        name="报告撰写者",
        system_prompt=(
            "你是一个报告撰写者。\n"
            "1. 接收信息收集者提供的要点\n"
            "2. 将这些要点整理成一篇完整的技术文章\n"
            "3. 文章要有引言、正文和总结\n"
            "用中文回答。"
        ),
    )

    # 用户输入
    user_input = "MCP 协议的优势和应用场景"
    print(f"\n[用户输入] {user_input}\n")

    # Agent A 收集信息
    print("─" * 40)
    print("[Agent A] 信息收集中...")
    print("─" * 40)
    result_a = await agent_a.reply(
        UserMsg(name="user", content=f"请搜索 '{user_input}' 的相关信息，整理成要点列表")
    )
    print(f"\n[Agent A 输出]\n{result_a.content}\n")

    # Agent A 将结果发送给 Agent B (A2A 核心!)
    print("─" * 40)
    print("[A2A] Agent A → Agent B 传递消息...")
    print("─" * 40)
    msg_to_b = UserMsg(
        name="信息收集者",  # 发送者名称
        content=(
            f"这是关于 '{user_input}' 的信息要点，请基于这些要点撰写一篇技术文章:\n\n"
            f"{result_a.content}"
        ),
    )

    # Agent B 基于 Agent A 的结果撰写报告
    result_b = await agent_b.reply(msg_to_b)
    print(f"\n[Agent B 最终报告]\n{result_b.content}\n")

    return result_b.content


# ============================================================
# 主程序
# ============================================================
async def main():
    print("A2A (Agent-to-Agent) 协议演示")
    print("=" * 60)

    # 方式 1: 串行 Pipeline
    await serial_pipeline()

    # 方式 2: 并行 Pipeline
    await parallel_pipeline()

    # 方式 3: Agent 间直接消息传递
    await agent_to_agent_messaging()

    # 总结
    print("\n" + "=" * 60)
    print("[A2A 总结]")
    print("=" * 60)
    print("""
A2A (Agent-to-Agent) 三种实现方式:

1. 串行 Pipeline
   Agent A → Agent B → Agent C
   适用: 需要多步骤处理的任务 (分析→翻译→校对)

2. 并行 Pipeline
   Agent A ┐
   Agent B ├→ 合并结果
   Agent C ┘
   适用: 可以同时处理的独立子任务

3. 直接消息传递
   Agent A 完成 → 消息 → Agent B 继续
   适用: 需要上下游协作的任务

关键点:
- Agent 间通过 UserMsg 传递消息
- 消息的 name 字段标识发送者
- Agent 的 system_prompt 定义其角色和职责
- asyncio.gather() 实现并行执行
    """)


if __name__ == "__main__":
    asyncio.run(main())
