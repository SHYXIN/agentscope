# -*- coding: utf-8 -*-
"""
实战 6: MCP 工具集成
====================
MCP (Model Context Protocol) 是标准化 LLM 与外部工具的通信协议。

本实战演示两种 MCP 集成方式:
  1. 模拟 MCP 工具 (不依赖外部 MCP Server，快速体验)
  2. 连接真实 MCP Server (需要安装 mcp-server-filesystem)

MCP 核心概念:
  - MCPClient: 连接到 MCP Server 的客户端
  - StdioMCPConfig: 通过 stdio 启动本地 MCP Server
  - HttpMCPConfig: 通过 HTTP 连接远程 MCP Server
  - MCPTool: 从 MCP Server 获取的工具，和原生 ToolBase 一样使用

运行方式:
    D:\code_project\github-proj\agentscope\.venv\Scripts\python.exe learn\06_mcp_integration.py
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
# 模拟 MCP 工具: 文件系统工具
# (模拟从 MCP Server 获取的工具，不依赖外部服务)
# ============================================================
class MCPFileReadTool(ToolBase):
    """
    模拟 MCP 文件读取工具
    对应 MCP Server 的 read_file 工具
    """

    name = "mcp_read_file"
    description = "读取指定路径的文件内容。"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径，如 '/home/user/readme.txt'",
            }
        },
        "required": ["path"],
    }
    is_concurrency_safe = True
    is_read_only = True
    is_mcp = True  # 标记为 MCP 工具
    mcp_name = "filesystem"  # 来源 MCP Server

    # 模拟文件系统
    _mock_files = {
        "/home/user/readme.txt": "这是 readme 文件的内容。\n欢迎使用 AgentScope！",
        "/home/user/config.json": '{"model": "LongCat-2.0-Preview", "stream": true}',
        "/home/user/data.csv": "name,age,city\nAlice,30,Beijing\nBob,25,Shanghai",
        "/tmp/test.txt": "这是一个临时文件。",
    }

    async def check_permissions(self, tool_input, context):
        return PermissionDecision(
            behavior=PermissionBehavior.ALLOW,
            message="文件读取是安全的",
        )

    async def __call__(self, path: str, **kwargs) -> ToolChunk:
        content = self._mock_files.get(path)
        if content:
            return ToolChunk(content=[TextBlock(text=f"文件内容 ({path}):\n{content}")])
        return ToolChunk(content=[TextBlock(text=f"文件不存在: {path}")])


class MCPFileWriteTool(ToolBase):
    """
    模拟 MCP 文件写入工具
    对应 MCP Server 的 write_file 工具
    """

    name = "mcp_write_file"
    description = "写入内容到指定路径的文件。"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径",
            },
            "content": {
                "type": "string",
                "description": "要写入的内容",
            },
        },
        "required": ["path", "content"],
    }
    is_concurrency_safe = False
    is_read_only = False
    is_mcp = True
    mcp_name = "filesystem"

    _mock_files: dict[str, str] = {}

    async def check_permissions(self, tool_input, context):
        return PermissionDecision(
            behavior=PermissionBehavior.ALLOW,
            message="文件写入已确认",
        )

    async def __call__(self, path: str, content: str, **kwargs) -> ToolChunk:
        self._mock_files[path] = content
        return ToolChunk(content=[
            TextBlock(text=f"已写入 {len(content)} 个字符到 {path}")
        ])


class MCPFileSearchTool(ToolBase):
    """
    模拟 MCP 文件搜索工具
    对应 MCP Server 的 search_files 工具
    """

    name = "mcp_search_files"
    description = "在指定目录中搜索文件。"
    input_schema = {
        "type": "object",
        "properties": {
            "directory": {
                "type": "string",
                "description": "要搜索的目录路径",
            },
            "pattern": {
                "type": "string",
                "description": "搜索模式，如 '*.txt' 或 '*.py'",
            },
        },
        "required": ["directory", "pattern"],
    }
    is_concurrency_safe = True
    is_read_only = True
    is_mcp = True
    mcp_name = "filesystem"

    async def check_permissions(self, tool_input, context):
        return PermissionDecision(
            behavior=PermissionBehavior.ALLOW,
            message="文件搜索是安全的",
        )

    async def __call__(self, directory: str, pattern: str, **kwargs) -> ToolChunk:
        # 模拟搜索结果
        mock_results = {
            ("/home/user", "*.txt"): ["readme.txt", "notes.txt"],
            ("/home/user", "*.json"): ["config.json"],
            ("/home/user", "*.csv"): ["data.csv"],
            ("/tmp", "*.txt"): ["test.txt"],
            ("/tmp", "*"): ["test.txt", "cache.dat"],
        }
        key = (directory, pattern)
        files = mock_results.get(key, [])
        if files:
            return ToolChunk(content=[
                TextBlock(text=f"在 {directory} 中搜索 '{pattern}' 的结果:\n" + "\n".join(f"  - {f}" for f in files))
            ])
        return ToolChunk(content=[
            TextBlock(text=f"在 {directory} 中未找到匹配 '{pattern}' 的文件")
        ])


# ============================================================
# 模拟 MCP 工具: 天气服务
# ============================================================
class MCPWeatherTool(ToolBase):
    """
    模拟 MCP 天气服务工具
    对应远程 MCP Server 的 get_weather 工具
    """

    name = "mcp_get_weather"
    description = "获取指定城市的实时天气信息。"
    input_schema = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称",
            }
        },
        "required": ["city"],
    }
    is_concurrency_safe = True
    is_read_only = True
    is_mcp = True
    mcp_name = "weather-service"

    _weather_data = {
        "北京": {"temp": 25, "condition": "晴", "humidity": 45},
        "上海": {"temp": 28, "condition": "多云", "humidity": 65},
        "广州": {"temp": 32, "condition": "小雨", "humidity": 80},
        "深圳": {"temp": 30, "condition": "阴", "humidity": 70},
        "杭州": {"temp": 26, "condition": "晴", "humidity": 55},
    }

    async def check_permissions(self, tool_input, context):
        return PermissionDecision(
            behavior=PermissionBehavior.ALLOW,
            message="天气查询是安全的",
        )

    async def __call__(self, city: str, **kwargs) -> ToolChunk:
        data = self._weather_data.get(city)
        if data:
            return ToolChunk(content=[TextBlock(
                text=f"{city}天气: {data['condition']}, "
                     f"温度 {data['temp']}°C, 湿度 {data['humidity']}%"
            )])
        return ToolChunk(content=[TextBlock(text=f"未找到 {city} 的天气数据")])


# ============================================================
# 辅助函数: 流式输出
# ============================================================
async def stream_reply(agent: Agent, user_msg: UserMsg, label: str = ""):
    """流式输出 Agent 回复"""
    if label:
        print(f"\n[{label}]")

    tool_args_buffer = ""
    tool_result_buffer = ""

    async for event in agent.reply_stream(user_msg):
        event_type = type(event).__name__

        if event_type == "TextBlockDeltaEvent":
            print(event.delta, end="", flush=True)

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

    # ========================================
    # 场景 1: 混合使用原生工具和 MCP 工具
    # ========================================
    print("=" * 60)
    print("[场景 1] 混合使用原生工具和 MCP 工具")
    print("=" * 60)

    # 创建混合工具包
    toolkit = Toolkit(
        tools=[
            # 原生工具
            MCPFileReadTool(),      # 模拟 MCP 文件读取
            MCPFileWriteTool(),     # 模拟 MCP 文件写入
            MCPFileSearchTool(),    # 模拟 MCP 文件搜索
            MCPWeatherTool(),       # 模拟 MCP 天气服务
        ]
    )

    agent = Agent(
        name="MCP-Demo",
        system_prompt=(
            "你是一个文件管理助手。你可以使用以下工具:\n"
            "- mcp_read_file: 读取文件内容\n"
            "- mcp_write_file: 写入文件\n"
            "- mcp_search_files: 搜索文件\n"
            "- mcp_get_weather: 查询天气\n"
            "\n"
            "注意: mcp_read_file 和 mcp_get_weather 是只读工具，直接执行。\n"
            "mcp_write_file 是写操作。\n"
            "用中文回答。"
        ),
        model=model,
        toolkit=toolkit,
        react_config=ReActConfig(max_iters=10),
    )

    # 测试 1: 读取文件
    await stream_reply(
        agent,
        UserMsg(name="user", content="请读取 /home/user/readme.txt 的内容"),
        label="测试 1: 读取文件",
    )

    # 测试 2: 搜索文件
    await stream_reply(
        agent,
        UserMsg(name="user", content="在 /home/user 目录下搜索所有 .txt 文件"),
        label="测试 2: 搜索文件",
    )

    # 测试 3: 写入文件
    await stream_reply(
        agent,
        UserMsg(name="user", content="在 /home/user/notes.txt 中写入 '今天学习了 MCP 协议'"),
        label="测试 3: 写入文件",
    )

    # 测试 4: 查询天气
    await stream_reply(
        agent,
        UserMsg(name="user", content="北京今天天气怎么样?"),
        label="测试 4: 查询天气",
    )

    # ========================================
    # 场景 2: 展示 MCP 工具属性
    # ========================================
    print("\n" + "=" * 60)
    print("[场景 2] MCP 工具属性")
    print("=" * 60)

    # 直接从我们创建的工具实例打印属性
    from agentscope.tool import ToolBase
    all_tools = [MCPFileReadTool(), MCPFileWriteTool(), MCPFileSearchTool(), MCPWeatherTool()]
    for tool in all_tools:
        print(f"\n工具: {tool.name}")
        print(f"  is_mcp: {tool.is_mcp}")
        print(f"  mcp_name: {tool.mcp_name}")
        print(f"  is_read_only: {tool.is_read_only}")
        print(f"  is_concurrency_safe: {tool.is_concurrency_safe}")

    # ========================================
    # 场景 3: 连接真实 MCP Server (可选)
    # ========================================
    print("\n" + "=" * 60)
    print("[场景 3] 连接真实 MCP Server")
    print("=" * 60)
    print("""
要连接真实的 MCP Server，需要:

1. 安装 MCP Server (例如 filesystem server):
   pip install mcp-server-filesystem

2. 创建 MCPClient:
   from agentscope.mcp import MCPClient, StdioMCPConfig

   mcp_client = MCPClient(
       name="filesystem",
       is_stateful=True,
       mcp_config=StdioMCPConfig(
           command="mcp-server-filesystem",
           args=["/path/to/allowed/directory"],
       ),
   )

3. 注册到 Toolkit:
   toolkit = Toolkit(mcps=[mcp_client])

4. Agent 就能使用 MCP Server 提供的工具了

HTTP MCP Server 示例:
   mcp_client = MCPClient(
       name="weather",
       is_stateful=False,
       mcp_config=HttpMCPConfig(
           url="https://api.weather.com/mcp",
       ),
   )
    """)


if __name__ == "__main__":
    asyncio.run(main())
