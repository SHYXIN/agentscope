# -*- coding: utf-8 -*-
"""
实战 7 (客户端): 连接真实 MCP Server
=====================================
连接到 07_mcp_server.py 启动的 MCP Server，使用 MCP 工具完成复杂任务。

运行方式:
    先在一个终端启动 MCP Server:
        .venv\Scripts\python.exe learn\07_mcp_server.py

    再在另一个终端运行本文件:
        .venv\Scripts\python.exe learn\07_mcp_agent.py
"""

import os
import sys
import asyncio
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
from agentscope.mcp import MCPClient, HttpMCPConfig
from agentscope.tool import Toolkit, ToolBase, ToolChunk
from agentscope.permission import (
    PermissionDecision,
    PermissionBehavior,
    PermissionContext,
)
from agentscope.message import TextBlock, UserMsg


# ============================================================
# 自定义 MCP Server 地址
# ============================================================
MCP_SERVER_URL = "http://127.0.0.1:8765/mcp"


# ============================================================
# 辅助函数: 流式输出
# ============================================================
async def stream_reply(agent: Agent, user_msg: UserMsg, label: str = ""):
    """流式输出 Agent 回复，展示 ReAct 每一步"""
    if label:
        print(f"\n[{label}]")

    tool_args_buffer = ""
    tool_result_buffer = ""

    async for event in agent.reply_stream(user_msg):
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
            print("\n  [回复完成]")

    print()


# ============================================================
# 主程序
# ============================================================
async def main():
    # 1. 创建 MCP Client 连接到 MCP Server
    print(f"[MCP] 连接到 MCP Server: {MCP_SERVER_URL}")

    mcp_client = MCPClient(
        name="my-tools",
        is_stateful=False,  # 无状态模式，每次调用创建新连接
        mcp_config=HttpMCPConfig(
            url=MCP_SERVER_URL,
            timeout=30.0,
        ),
    )

    # 2. 列出 MCP Server 提供的所有工具
    print("\n[MCP] 可用工具列表:")
    raw_tools = await mcp_client.list_raw_tools()
    for tool in raw_tools:
        schema = tool.inputSchema
        props = schema.get("properties", {})
        required = schema.get("required", [])
        print(f"  - {tool.name}: {tool.description}")
        print(f"    参数: {list(props.keys())}")
        print(f"    必填: {required}")

    # 3. 将 MCP 工具注册到 Toolkit
    toolkit = Toolkit(mcps=[mcp_client])

    # 4. 创建 Agent
    agent = Agent(
        name="MCP-Agent",
        system_prompt=(
            "你是一个智能助手，可以通过 MCP 协议调用外部工具。\n"
            "可用工具:\n"
            "- mcp__demo-server__get_weather: 查询城市天气\n"
            "- mcp__demo-server__translate_text: 翻译文本\n"
            "- mcp__demo-server__calculate: 执行数学计算\n"
            "请根据用户需求合理使用工具，用中文回答。"
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
        toolkit=toolkit,
        react_config=ReActConfig(max_iters=10),
    )

    # 5. 测试: 使用 MCP 工具
    print("\n" + "=" * 60)
    print("[测试 1] 查询天气 (MCP 工具)")
    print("=" * 60)
    await stream_reply(
        agent,
        UserMsg(name="user", content="北京今天天气怎么样?"),
    )

    print("\n" + "=" * 60)
    print("[测试 2] 翻译 (MCP 工具)")
    print("=" * 60)
    await stream_reply(
        agent,
        UserMsg(name="user", content="请用英文翻译 '你好世界'"),
    )

    print("\n" + "=" * 60)
    print("[测试 3] 数学计算 (MCP 工具)")
    print("=" * 60)
    await stream_reply(
        agent,
        UserMsg(name="user", content="计算 (3.14 * 25) + 100 等于多少?"),
    )

    print("\n" + "=" * 60)
    print("[测试 4] 组合使用多个 MCP 工具")
    print("=" * 60)
    await stream_reply(
        agent,
        UserMsg(name="user", content="上海的天气怎么样? 如果温度大于 28 度，建议穿短袖。请用中英文给出建议。"),
    )


if __name__ == "__main__":
    asyncio.run(main())
