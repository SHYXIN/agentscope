# -*- coding: utf-8 -*-
"""
实战 2: 中间件机制 (Middleware)
===============================
目标: 理解 AgentScope 的洋葱模式中间件系统

中间件可以拦截 5 个关键执行点:
  1. on_reply     — 整个回复过程
  2. on_reasoning — 推理/模型调用阶段
  3. on_acting    — 工具执行阶段
  4. on_model_call — 原始模型 API 调用
  5. on_system_prompt — 系统提示词转换

本实战实现 3 个中间件:
  - LoggingMiddleware: 记录每个阶段的耗时
  - RateLimitMiddleware: 限制工具调用频率
  - PromptEnhancerMiddleware: 动态增强 system prompt

运行方式:
    D:\code_project\github-proj\agentscope\.venv\Scripts\python.exe learn\02_middleware.py
"""

import os
import sys
import asyncio
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Awaitable, Union
from datetime import datetime

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
from agentscope.middleware import MiddlewareBase
from agentscope.model import ChatResponse


# ============================================================
# 中间件 1: 日志记录 — 记录每个阶段的耗时
# ============================================================
class LoggingMiddleware(MiddlewareBase):
    """
    日志中间件: 在推理和工具调用前后打印日志

    展示了洋葱模式的核心用法:
    - next_handler() 之前 = 前置逻辑
    - next_handler() 之后 = 后置逻辑
    """

    async def on_reasoning(
        self,
        agent: "Agent",
        input_kwargs: dict,
        next_handler: Callable[..., AsyncGenerator],
    ) -> AsyncGenerator:
        """拦截推理阶段，记录耗时"""
        start = time.time()
        print(f"  [日志] 推理开始 | 时间: {datetime.now().strftime('%H:%M:%S')}")

        # 调用下一个中间件或原始推理逻辑
        async for event in next_handler(**input_kwargs):
            yield event

        elapsed = time.time() - start
        print(f"  [日志] 推理结束 | 耗时: {elapsed:.2f}s")
        # 注意: yield 之后的代码在所有事件都发送完之后才执行

    async def on_acting(
        self,
        agent: "Agent",
        input_kwargs: dict,
        next_handler: Callable[..., AsyncGenerator],
    ) -> AsyncGenerator:
        """拦截工具调用，记录工具名和耗时"""
        tool_call = input_kwargs.get("tool_call")
        tool_name = tool_call.name if hasattr(tool_call, "name") else "unknown"

        start = time.time()
        print(f"  [日志] 调用工具: {tool_name}")

        async for event in next_handler(**input_kwargs):
            yield event

        elapsed = time.time() - start
        print(f"  [日志] 工具返回: {tool_name} | 耗时: {elapsed:.2f}s")


# ============================================================
# 中间件 2: 频率限制 — 限制工具调用频率
# ============================================================
class RateLimitMiddleware(MiddlewareBase):
    """
    频率限制中间件: 限制每个工具在指定时间窗口内的调用次数

    这是中间件拦截 acting 阶段的典型用法。
    在生产环境中，这可以防止 Agent 滥用外部 API。
    """

    def __init__(self, max_calls: int = 5, window_seconds: float = 60.0):
        """
        Args:
            max_calls: 时间窗口内最大调用次数
            window_seconds: 时间窗口大小（秒）
        """
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        # 记录每个工具的调用时间戳 {tool_name: [timestamp, ...]}
        self._call_history: dict[str, list[float]] = {}

    async def on_acting(
        self,
        agent: "Agent",
        input_kwargs: dict,
        next_handler: Callable[..., AsyncGenerator],
    ) -> AsyncGenerator:
        """检查频率限制，超限则拒绝执行"""
        tool_call = input_kwargs.get("tool_call")
        tool_name = tool_call.name if hasattr(tool_call, "name") else "unknown"

        now = time.time()

        # 初始化调用历史
        if tool_name not in self._call_history:
            self._call_history[tool_name] = []

        # 清理过期的调用记录
        self._call_history[tool_name] = [
            t for t in self._call_history[tool_name]
            if now - t < self.window_seconds
        ]

        # 检查是否超限
        if len(self._call_history[tool_name]) >= self.max_calls:
            print(f"  [限流] 工具 '{tool_name}' 调用频率超限 "
                  f"({self.max_calls}次/{self.window_seconds}s)，拒绝执行！")
            # 不调用 next_handler，直接返回
            # 但需要返回一个错误结果给 Agent
            return

        # 记录本次调用
        self._call_history[tool_name].append(now)
        remaining = self.max_calls - len(self._call_history[tool_name])
        print(f"  [限流] 工具 '{tool_name}' 调用通过 | 剩余次数: {remaining}")

        # 正常执行
        async for event in next_handler(**input_kwargs):
            yield event


# ============================================================
# 中间件 3: Prompt 增强 — 动态修改系统提示词
# ============================================================
class PromptEnhancerMiddleware(MiddlewareBase):
    """
    Prompt 增强中间件: 在运行时动态修改 system prompt

    这是 Transformer 模式（而非洋葱模式）的典型用法。
    多个 PromptEnhancer 会形成处理链，每个都修改上一步的输出。

    典型用途:
    - 注入当前时间、用户信息等动态内容
    - 根据上下文动态调整 Agent 行为
    - 添加安全提示或合规要求
    """

    async def on_system_prompt(
        self,
        agent: "Agent",
        current_prompt: str,
    ) -> str:
        """在原始 prompt 基础上追加动态内容"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        enhancement = (
            f"\n\n--- 动态信息 ---\n"
            f"当前时间: {now}\n"
            f"Agent 名称: {agent.name}\n"
            f"会话 ID: {agent.state.session_id[:8]}...\n"
            f"----------------"
        )
        enhanced = current_prompt + enhancement
        print(f"  [Prompt] 已注入动态信息 (当前时间: {now})")
        return enhanced


# ============================================================
# 工具定义 (复用实战 1 的工具)
# ============================================================
class CalculatorTool(ToolBase):
    name = "calculator"
    description = "执行数学计算。支持加减乘除和幂运算。"
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "数学表达式"}
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
            return ToolChunk(content=[TextBlock(text=f"计算结果: {result}")])
        except Exception as e:
            return ToolChunk(content=[TextBlock(text=f"计算错误: {e}")])


class WeatherTool(ToolBase):
    name = "get_weather"
    description = "查询指定城市的天气。"
    input_schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名称"}
        },
        "required": ["city"],
    }
    is_concurrency_safe = True
    is_read_only = True

    _weather_data = {
        "北京": {"temp": 25, "condition": "晴"},
        "上海": {"temp": 28, "condition": "多云"},
        "广州": {"temp": 32, "condition": "小雨"},
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
                TextBlock(text=f"{city}: {data['temp']}C, {data['condition']}")
            ])
        return ToolChunk(content=[TextBlock(text=f"未找到 {city} 的天气")])


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
    toolkit = Toolkit(tools=[CalculatorTool(), WeatherTool()])

    # 3. 创建 3 个中间件
    logging_mw = LoggingMiddleware()
    rate_limit_mw = RateLimitMiddleware(max_calls=3, window_seconds=60)
    prompt_mw = PromptEnhancerMiddleware()

    # 4. 创建 Agent（挂载中间件）
    agent = Agent(
        name="MiddlewareDemo",
        system_prompt=(
            "你是一个 AI 助手。可以使用以下工具:\n"
            "- calculator: 执行数学计算\n"
            "- get_weather: 查询城市天气\n"
            "用中文回答。"
        ),
        model=model,
        toolkit=toolkit,
        react_config=ReActConfig(max_iters=10),
        # 中间件列表: 按顺序形成洋葱层
        # 执行顺序: logging -> rate_limit -> prompt -> agent_core
        # 返回顺序: agent_core -> prompt -> rate_limit -> logging
        middlewares=[logging_mw, rate_limit_mw, prompt_mw],
    )

    # 5. 测试: 展示中间件效果
    test_questions = [
        "计算 2 的 10 次方是多少?",
        "北京天气怎么样?",
        "再算一下 100 除以 7 等于多少?",
    ]

    for i, question in enumerate(test_questions, 1):
        print("=" * 60)
        print(f"[测试 {i}] 用户: {question}")
        print("=" * 60)

        tool_args_buffer = ""
        tool_result_buffer = ""

        async for event in agent.reply_stream(
            UserMsg(name="user", content=question)
        ):
            event_type = type(event).__name__

            if event_type == "TextBlockDeltaEvent":
                print(event.delta, end="", flush=True)

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
                print("\n\n[回复完成]")

        print()


if __name__ == "__main__":
    asyncio.run(main())
