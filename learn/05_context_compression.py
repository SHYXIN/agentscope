# -*- coding: utf-8 -*-
"""
实战 5: 上下文压缩 (Context Compression)
=========================================
场景: 当对话历史太长，超过模型的 context window 时，
      AgentScope 会自动将旧对话压缩成摘要。

原理:
  1. 每次推理前，Agent 会估算当前上下文的 token 数
  2. 如果超过 trigger_ratio (默认 0.8) * context_size，触发压缩
  3. 将旧消息送给 LLM，生成结构化摘要 (SummarySchema)
  4. 用摘要替换旧消息，保留最近的对话

运行方式:
    D:\code_project\github-proj\agentscope\.venv\Scripts\python.exe learn\05_context_compression.py
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
from agentscope.agent._config import ReActConfig, ContextConfig
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
# 工具: 知识搜索
# ============================================================
class SearchTool(ToolBase):
    name = "search"
    description = "搜索知识库获取信息。"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"}
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
            "python": "Python 是一种高级编程语言，广泛用于 AI、数据科学和 Web 开发。",
            "react": "ReAct = Reasoning + Acting，Agent 交替进行思考和行动。",
            "rag": "RAG = Retrieval-Augmented Generation，先检索再生成。",
            "mcp": "MCP = Model Context Protocol，标准化 LLM 与工具的通信协议。",
            "agent": "Agent 是能感知环境、自主决策的系统，核心能力包括推理、规划、工具使用。",
            "llm": "LLM = Large Language Model，大语言模型，如 GPT、Claude、LongCat。",
            "transformer": "Transformer 是 LLM 的核心架构，基于自注意力机制。",
            "embedding": "Embedding 是将文本转换为向量表示的技术。",
            "vector": "向量数据库用于存储和检索 embedding，支持语义搜索。",
            "prompt": "Prompt 是给 LLM 的输入指令，好的 prompt 能显著提升输出质量。",
        }
        for key, value in knowledge.items():
            if key in query.lower():
                return ToolChunk(content=[TextBlock(text=value)])
        return ToolChunk(content=[TextBlock(text=f"未找到 '{query}' 的信息")])


# ============================================================
# 辅助函数: 流式输出 + 打印上下文信息
# ============================================================
async def stream_and_observe(agent: Agent, user_msg: UserMsg, label: str = ""):
    """流式输出，并打印上下文变化"""
    if label:
        print(f"\n[{label}]")

    # 打印压缩前的上下文信息
    context_before = len(agent.state.context)
    has_summary_before = bool(agent.state.summary)
    print(f"  [上下文] 消息数: {context_before}, 摘要: {'有' if has_summary_before else '无'}")

    full_text = ""
    tool_args_buffer = ""
    tool_result_buffer = ""
    compression_happened = False

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
            print("\n  [回复完成]")

    # 打印压缩后的上下文信息
    context_after = len(agent.state.context)
    has_summary_after = bool(agent.state.summary)
    print(f"  [上下文] 消息数: {context_after}, 摘要: {'有' if has_summary_after else '无'}")

    if has_summary_after and not has_summary_before:
        print(f"\n  [!] 上下文压缩已触发！")
        print(f"  [摘要内容] {agent.state.summary[:200]}...")

    return full_text


# ============================================================
# 主程序
# ============================================================
async def main():
    # 1. 创建模型
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

    # 2. 创建工具包
    toolkit = Toolkit(tools=[SearchTool()])

    # 3. 创建 Agent
    # 使用默认的 ContextConfig: trigger_ratio=0.8, reserve_ratio=0.1
    agent = Agent(
        name="压缩演示",
        system_prompt=(
            "你是一个知识助手。使用 search 工具搜索知识库回答问题。"
            "用中文回答，回答要详细。"
        ),
        model=model,
        toolkit=toolkit,
        react_config=ReActConfig(max_iters=10),
        # ContextConfig 使用默认值:
        # trigger_ratio=0.8  (超过 80% context 时触发压缩)
        # reserve_ratio=0.1  (保留 10% 给最近消息)
    )

    # ========================================
    # 阶段 1: 正常对话 (不触发压缩)
    # ========================================
    print("=" * 60)
    print("[阶段 1] 正常对话 - 上下文较短，不触发压缩")
    print("=" * 60)

    questions_short = [
        "什么是 Python?",
        "什么是 ReAct?",
    ]

    for q in questions_short:
        print(f"\n用户: {q}")
        await stream_and_observe(
            agent, UserMsg(name="user", content=q), label="Agent 回复"
        )

    # ========================================
    # 阶段 2: 长对话 (触发压缩)
    # ========================================
    print("\n" + "=" * 60)
    print("[阶段 2] 长对话 - 上下文增长，观察压缩触发")
    print("=" * 60)

    # 用很多问题让上下文增长
    questions_long = [
        "什么是 RAG? 请详细解释。",
        "什么是 MCP 协议? 它有什么优势?",
        "什么是 Agent? 它的核心能力是什么?",
        "什么是 LLM? 它和传统 AI 有什么区别?",
        "什么是 Transformer 架构?",
        "什么是 Embedding? 它在 AI 中有什么作用?",
        "什么是向量数据库? 它和传统数据库有什么区别?",
        "什么是 Prompt Engineering? 如何写好 prompt?",
        "Python 和 Java 有什么区别? 各自适合什么场景?",
        "ReAct 和 RAG 如何结合使用?",
    ]

    for i, q in enumerate(questions_long, 1):
        print(f"\n{'─'*40}")
        print(f"[轮次 {i}] 用户: {q}")
        await stream_and_observe(
            agent, UserMsg(name="user", content=q), label="Agent 回复"
        )

        # 每 3 轮打印一次上下文状态
        if i % 3 == 0:
            print(f"\n  [状态] 上下文消息数: {len(agent.state.context)}, "
                  f"摘要: {'有' if agent.state.summary else '无'}")

    # ========================================
    # 阶段 3: 验证压缩后 Agent 仍能记住关键信息
    # ========================================
    print("\n" + "=" * 60)
    print("[阶段 3] 验证 - 压缩后 Agent 是否仍记得之前讨论的内容")
    print("=" * 60)

    # 问一个之前讨论过的问题，看 Agent 是否能回答
    review_question = "我们之前讨论过 ReAct，请简要回顾一下它的核心思想。"
    print(f"\n用户: {review_question}")
    await stream_and_observe(
        agent, UserMsg(name="user", content=review_question), label="Agent 回复"
    )

    # ========================================
    # 总结
    # ========================================
    print("\n" + "=" * 60)
    print("[上下文压缩总结]")
    print("=" * 60)
    print(f"""
上下文压缩机制:

  触发条件:
    - 当前 token 数 > trigger_ratio (0.8) * context_size
    - 默认在 80% 时触发

  压缩过程:
    1. 将旧消息 (超出 reserve_ratio 的部分) 送给 LLM
    2. LLM 生成结构化摘要 (SummarySchema):
       - task_overview: 用户的核心需求
       - current_state: 目前完成了什么
       - important_discoveries: 关键发现
       - next_steps: 下一步要做什么
       - context_to_preserve: 需要保留的用户偏好
    3. 用摘要替换旧消息，保留最近的消息

  最终状态:
    - 上下文消息数: {len(agent.state.context)}
    - 摘要: {'有' if agent.state.summary else '无'}
    - 摘要长度: {len(agent.state.summary) if agent.state.summary else 0} 字符

  关键配置 (ContextConfig):
    - trigger_ratio: 0.8  (触发压缩的阈值)
    - reserve_ratio: 0.1  (保留给最近消息的比例)
    - compression_prompt: 引导 LLM 生成摘要的提示词
    - summary_schema: 摘要的结构化模型
    """)


if __name__ == "__main__":
    asyncio.run(main())
