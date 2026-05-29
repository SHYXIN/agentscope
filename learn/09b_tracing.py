# -*- coding: utf-8 -*-
"""
实战 9b: Tracing — OpenTelemetry 分布式追踪
=============================================
TracingMiddleware 自动记录 Agent 的每次推理、工具调用、Token 使用等。

本实战演示:
  1. 自定义简易追踪中间件（不依赖 OpenTelemetry SDK）
  2. 追踪每次推理的耗时
  3. 追踪工具调用的输入输出
  4. 追踪 Token 使用情况

运行方式:
    .venv\Scripts\python.exe learn\09b_tracing.py
"""

import os
import sys
import asyncio
import time
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
    ToolCallEndEvent,
    ToolResultEndEvent,
    ReplyStartEvent,
    ReplyEndEvent,
)


# ============================================================
# 自定义简易追踪中间件
# ============================================================
class SimpleTracingMiddleware(MiddlewareBase):
    """简易追踪中间件 — 记录每个阶段的耗时和事件"""

    def __init__(self):
        self.logs = []
        self._reply_start = None

    def _log(self, msg: str):
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {msg}"
        self.logs.append(entry)
        print(entry)

    async def on_reply(self, agent, input_kwargs, next_handler):
        """追踪整个回复过程"""
        self._log(f"=== 回复开始 | Agent: {agent.name} ===")
        self._reply_start = time.time()

        async for event in next_handler(**input_kwargs):
            event_type = type(event).__name__

            if isinstance(event, ModelCallStartEvent):
                self._log(f"  [推理开始] 模型: {event.model_name}")

            elif isinstance(event, ModelCallEndEvent):
                self._log(
                    f"  [推理结束] 输入tokens: {event.input_tokens}, "
                    f"输出tokens: {event.output_tokens}"
                )

            elif isinstance(event, ToolCallStartEvent):
                self._log(f"  [工具调用] {event.tool_call_name}")

            elif isinstance(event, ToolCallEndEvent):
                self._log(f"  [工具参数] 传输完成")

            elif isinstance(event, ToolResultEndEvent):
                state = str(event.state)
                status = "成功" if "running" in state else f"失败({state})"
                self._log(f"  [工具返回] 状态: {status}")

            elif isinstance(event, ReplyEndEvent):
                elapsed = time.time() - self._reply_start
                self._log(f"  [回复结束] 耗时: {elapsed:.2f}s")

            yield event

        elapsed = time.time() - self._reply_start
        self._log(f"=== 回复完成 | 总耗时: {elapsed:.2f}s ===\n")

    async def on_reasoning(self, agent, input_kwargs, next_handler):
        """追踪推理阶段"""
        self._log(f"  [阶段] 推理开始")
        start = time.time()

        async for event in next_handler(**input_kwargs):
            yield event

        elapsed = time.time() - start
        self._log(f"  [阶段] 推理结束 | 耗时: {elapsed:.2f}s")

    async def on_acting(self, agent, input_kwargs, next_handler):
        """追踪工具执行阶段"""
        tool_call = input_kwargs.get("tool_call")
        if tool_call:
            self._log(f"  [阶段] 执行工具: {tool_call.name}")
        start = time.time()

        async for event in next_handler(**input_kwargs):
            yield event

        elapsed = time.time() - start
        self._log(f"  [阶段] 工具执行结束 | 耗时: {elapsed:.2f}s")


# ============================================================
# 工具定义
# ============================================================
class SearchTool(ToolBase):
    name = "search"
    description = "搜索知识库。参数: query-搜索关键词"
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
            "agentscope": "AgentScope 是阿里巴巴开源的多智能体框架 2.0。",
        }
        for key, value in knowledge.items():
            if key in query.lower():
                return ToolChunk(content=[TextBlock(text=value)])
        return ToolChunk(content=[TextBlock(text=f"未找到 '{query}' 的信息")])


# ============================================================
# 主程序
# ============================================================
async def main():
    print("=" * 60)
    print("Tracing — 分布式追踪演示")
    print("=" * 60)

    # 创建带追踪中间件的 Agent
    tracer = SimpleTracingMiddleware()

    agent = Agent(
        name="Traced-Agent",
        system_prompt="你是一个智能助手。使用 search 工具搜索信息，用中文回答。",
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
        middlewares=[tracer],
    )

    # 测试问题
    questions = [
        "什么是 ReAct?",
        "什么是 RAG?",
        "MCP 协议有什么优势?",
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n{'─' * 40}")
        print(f"[测试 {i}] {q}")
        print("─" * 40)
        result = await agent.reply(UserMsg(name="user", content=q))
        print(f"\n[回答] {result.content}")

    # 打印追踪摘要
    print("\n" + "=" * 60)
    print("[追踪摘要]")
    print("=" * 60)
    print(f"共记录 {len(tracer.logs)} 条追踪日志")
    print("\n完整追踪日志:")
    for log in tracer.logs:
        print(f"  {log}")


if __name__ == "__main__":
    asyncio.run(main())
