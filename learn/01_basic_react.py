# -*- coding: utf-8 -*-
"""
实战 1: 最简 ReAct Agent
========================
目标: 理解 ReAct 的基本结构
- 注册 3 个工具: 计算器、天气查询、知识搜索
- 用 LongCat 模型驱动 Agent
- 观察 ReAct 循环的每一步

运行方式:
    D:\code_project\github-proj\agentscope\.venv\Scripts\python.exe learn\01_basic_react.py
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
# 工具 1: 计算器
# ============================================================
class CalculatorTool(ToolBase):
    """执行数学计算"""

    name = "calculator"
    description = (
        "执行数学计算。支持加减乘除和幂运算。"
        "例如: '25 * 4 + 100' 或 '2 ** 10'"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式, 如 '25 * 4 + 100'",
            }
        },
        "required": ["expression"],
    }
    is_concurrency_safe = True
    is_read_only = True

    async def check_permissions(self, tool_input, context):
        return PermissionDecision(
            behavior=PermissionBehavior.ALLOW,
            message="计算器是安全的，允许执行",
        )

    async def __call__(self, expression: str, **kwargs) -> ToolChunk:
        try:
            allowed = {"abs": abs, "max": max, "min": min, "round": round}
            result = eval(expression, {"__builtins__": {}}, allowed)
            return ToolChunk(content=[TextBlock(text=f"计算结果: {result}")])
        except Exception as e:
            return ToolChunk(content=[TextBlock(text=f"计算错误: {e}")])


# ============================================================
# 工具 2: 天气查询 (模拟)
# ============================================================
class WeatherTool(ToolBase):
    """查询城市天气"""

    name = "get_weather"
    description = "查询指定城市的天气。返回温度和天气状况。"
    input_schema = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称, 如 '北京'、'上海'",
            }
        },
        "required": ["city"],
    }
    is_concurrency_safe = True
    is_read_only = True

    _weather_data = {
        "北京": {"temp": 25, "condition": "晴"},
        "上海": {"temp": 28, "condition": "多云"},
        "广州": {"temp": 32, "condition": "小雨"},
        "深圳": {"temp": 30, "condition": "阴"},
        "杭州": {"temp": 26, "condition": "晴"},
    }

    async def check_permissions(self, tool_input, context):
        return PermissionDecision(
            behavior=PermissionBehavior.ALLOW,
            message="允许执行",
        )

    async def __call__(self, city: str, **kwargs) -> ToolChunk:
        data = self._weather_data.get(city)
        if data:
            return ToolChunk(content=[
                TextBlock(text=f"{city}: {data['temp']}°C, {data['condition']}")
            ])
        return ToolChunk(content=[
            TextBlock(text=f"未找到 {city} 的天气数据")
        ])


# ============================================================
# 工具 3: 知识搜索 (模拟 RAG)
# ============================================================
class SearchTool(ToolBase):
    """搜索知识库"""

    name = "search"
    description = (
        "搜索知识库获取信息。当需要查询某个概念、"
        "事实或知识点时使用此工具。"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词或问题",
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
                "ReAct = Reasoning + Acting (推理 + 行动)。"
                "Agent 交替进行思考和行动, 直到完成任务。"
                "这是 AgentScope 中最核心的 Agent 范式。"
            ),
            "rag": (
                "RAG = Retrieval-Augmented Generation (检索增强生成)。"
                "先检索外部知识, 再交给 LLM 生成回答。"
                "RAG 可以作为 ReAct Agent 的一个工具来使用。"
            ),
            "agentscope": (
                "AgentScope 是阿里巴巴开源的多智能体框架 2.0 版本。"
                "支持 ReAct、工具调用、MCP 协议、多智能体协作、"
                "Human-in-the-Loop、上下文压缩等核心功能。"
            ),
        }
        for key, value in knowledge.items():
            if key in query.lower():
                return ToolChunk(content=[TextBlock(text=value)])
        return ToolChunk(content=[
            TextBlock(text=f"未找到关于 '{query}' 的信息")
        ])


# ============================================================
# 主程序
# ============================================================
async def main():
    # 1. 创建模型 (LongCat 兼容 OpenAI 接口)
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
    toolkit = Toolkit(
        tools=[CalculatorTool(), WeatherTool(), SearchTool()]
    )

    # 3. 创建 ReAct Agent
    agent = Agent(
        name="小助手",
        system_prompt=(
            "你是一个友好的 AI 助手。你可以使用以下工具帮助用户:\n"
            "- calculator: 执行数学计算\n"
            "- get_weather: 查询城市天气\n"
            "- search: 搜索知识库获取信息\n"
            "请根据用户的需求合理使用工具, 用中文回答。"
        ),
        model=model,
        toolkit=toolkit,
        react_config=ReActConfig(max_iters=10),
    )

    # 4. 测试问题列表
    test_questions = [
        "请帮我计算 (3 + 5) * 2 等于多少? 分步计算。",
        "北京今天天气怎么样?",
        "什么是 ReAct? 请搜索知识库回答。",
    ]

    for i, question in enumerate(test_questions, 1):
        print("=" * 60)
        print(f"[测试 {i}] 用户: {question}")
        print("=" * 60)

        user_msg = UserMsg(name="user", content=question)

        # 流式输出, 观察 ReAct 的每一步
        print("\n[Agent 开始推理...]\n")

        # 用于拼接被流式切碎的文本
        tool_args_buffer = ""
        tool_result_buffer = ""

        async for event in agent.reply_stream(user_msg):
            event_type = type(event).__name__

            if event_type == "TextBlockDeltaEvent":
                # Agent 的文字回复增量
                print(event.delta, end="", flush=True)

            elif event_type == "ThinkingBlockDeltaEvent":
                # 思考过程增量 (如果模型支持)
                print(f"\n  [思考]: {event.delta}", end="", flush=True)

            elif event_type == "ToolCallStartEvent":
                # 工具调用开始 - 打印工具名
                print(f"\n  [调用工具] {event.tool_call_name}")
                tool_args_buffer = ""

            elif event_type == "ToolCallDeltaEvent":
                # 工具参数是流式传输的, 需要拼接
                tool_args_buffer += event.delta

            elif event_type == "ToolCallEndEvent":
                # 工具参数传输完毕, 打印完整参数
                print(f"  [工具参数] {tool_args_buffer}")
                tool_result_buffer = ""

            elif event_type == "ToolResultTextDeltaEvent":
                # 工具返回结果增量, 需要拼接
                tool_result_buffer += event.delta

            elif event_type == "ToolResultEndEvent":
                # 工具返回结果完毕
                print(f"  [工具返回] {tool_result_buffer}")

            elif event_type == "ReplyEndEvent":
                # 回复结束
                print("\n\n[回复完成]")

        print()


if __name__ == "__main__":
    asyncio.run(main())
