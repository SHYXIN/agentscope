# -*- coding: utf-8 -*-
"""
实战 4 (交互式): Human-in-the-Loop
====================================
体验 Agent 在执行危险操作前请求用户确认的流程

运行方式:
    D:\code_project\github-proj\agentscope\.venv\Scripts\python.exe learn\04_hitl_interactive.py
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
    PermissionRule,
)
from agentscope.message import (
    TextBlock,
    UserMsg,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultState,
)
from agentscope.event import (
    RequireUserConfirmEvent,
    UserConfirmResultEvent,
    ConfirmResult,
    ReplyEndEvent,
    TextBlockDeltaEvent,
    ToolCallStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolResultTextDeltaEvent,
    ToolResultEndEvent,
)


# ============================================================
# 工具 1: 计算器 (安全，直接执行)
# ============================================================
class CalculatorTool(ToolBase):
    name = "calculator"
    description = "执行数学计算。例如: '2 ** 10' 或 '100 / 7'"
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
            message="安全工具，直接执行",
        )

    async def __call__(self, expression: str, **kwargs) -> ToolChunk:
        try:
            allowed = {"abs": abs, "max": max, "min": min, "round": round}
            result = eval(expression, {"__builtins__": {}}, allowed)
            return ToolChunk(content=[TextBlock(text=f"结果: {result}")])
        except Exception as e:
            return ToolChunk(content=[TextBlock(text=f"错误: {e}")])


# ============================================================
# 工具 2: 删除文件 (危险，需要确认)
# ============================================================
class DeleteFileTool(ToolBase):
    name = "delete_file"
    description = (
        "【危险操作】删除指定路径的文件。"
        "此操作不可撤销，执行前会请求用户确认。"
        "参数: file_path - 要删除的文件路径"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "要删除的文件路径，如 '/tmp/test.txt'",
            }
        },
        "required": ["file_path"],
    }
    is_concurrency_safe = False
    is_read_only = False  # 写操作，危险

    async def check_permissions(self, tool_input, context):
        # 返回 ASK，触发 Human-in-the-Loop
        return PermissionDecision(
            behavior=PermissionBehavior.ASK,
            message=f"危险操作: 删除文件 {tool_input.get('file_path', '')}，需要用户确认",
        )

    async def __call__(self, file_path: str, **kwargs) -> ToolChunk:
        # 模拟删除（实际不会真正删除）
        return ToolChunk(content=[
            TextBlock(text=f"文件 '{file_path}' 已成功删除（模拟）")
        ])


# ============================================================
# 工具 3: 发送邮件 (危险，需要确认)
# ============================================================
class SendEmailTool(ToolBase):
    name = "send_email"
    description = (
        "【危险操作】发送邮件。"
        "参数: to-收件人, subject-主题, body-正文"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "收件人邮箱"},
            "subject": {"type": "string", "description": "邮件主题"},
            "body": {"type": "string", "description": "邮件正文"},
        },
        "required": ["to", "subject", "body"],
    }
    is_concurrency_safe = False
    is_read_only = False

    async def check_permissions(self, tool_input, context):
        return PermissionDecision(
            behavior=PermissionBehavior.ASK,
            message=f"危险操作: 发送邮件给 {tool_input.get('to', '')}，需要用户确认",
        )

    async def __call__(self, to: str, subject: str, body: str, **kwargs) -> ToolChunk:
        return ToolChunk(content=[
            TextBlock(text=f"邮件已发送给 {to}（模拟）\n主题: {subject}")
        ])


# ============================================================
# 辅助函数: 收集 Agent 的事件，检测是否需要确认
# ============================================================
async def run_agent_with_hitl(agent, user_msg):
    """
    运行 Agent，如果遇到 RequireUserConfirmEvent 就请求用户确认。
    返回最终回复消息。
    """
    pending_confirmations = []  # 待确认的工具调用
    collected_text = []         # 收集的文字输出

    async for event in agent.reply_stream(user_msg):
        event_type = type(event).__name__

        if event_type == "TextBlockDeltaEvent":
            print(event.delta, end="", flush=True)

        elif event_type == "ToolCallStartEvent":
            print(f"\n  [调用工具] {event.tool_call_name}")

        elif event_type == "ToolCallDeltaEvent":
            pass  # 参数增量，不打印

        elif event_type == "ToolCallEndEvent":
            pass  # 参数传输完毕

        elif event_type == "ToolResultTextDeltaEvent":
            print(f"  [工具返回] {event.delta}", end="", flush=True)

        elif event_type == "ToolResultEndEvent":
            pass

        elif event_type == "RequireUserConfirmEvent":
            # 收到确认请求！
            pending_confirmations.extend(event.tool_calls)
            print(f"\n{'='*50}")
            print(f"  ⚠️  需要用户确认: {len(event.tool_calls)} 个工具调用")
            for tc in event.tool_calls:
                print(f"    - {tc.name}: {tc.input}")
                if tc.suggested_rules:
                    print(f"      建议规则: {tc.suggested_rules}")
            print(f"{'='*50}")

        elif event_type == "ReplyEndEvent":
            print("\n  [回复完成]")

    return pending_confirmations


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
    toolkit = Toolkit(
        tools=[CalculatorTool(), DeleteFileTool(), SendEmailTool()]
    )

    # 3. 创建 Agent
    agent = Agent(
        name="助手",
        system_prompt=(
            "你是一个 AI 助手。你有以下工具:\n"
            "- calculator: 执行数学计算（安全）\n"
            "- delete_file: 删除文件（危险，会请求确认）\n"
            "- send_email: 发送邮件（危险，会请求确认）\n"
            "\n当用户要求删除文件或发送邮件时，直接调用对应工具。"
            "用中文回答。"
        ),
        model=model,
        toolkit=toolkit,
        react_config=ReActConfig(max_iters=10),
    )

    print("=" * 60)
    print("  Human-in-the-Loop 交互式演示")
    print("=" * 60)
    print()
    print("可用命令:")
    print("  1. 计算 <表达式>     - 安全操作，直接执行")
    print("  2. 删除 <文件路径>   - 危险操作，会请求确认")
    print("  3. 发送邮件          - 危险操作，会请求确认")
    print("  4. quit              - 退出")
    print()

    while True:
        user_input = input("你: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见!")
            break
        if not user_input:
            continue

        print(f"\n[Agent 思考中...]\n")

        # 运行 Agent，收集待确认事件
        pending = await run_agent_with_hitl(
            agent, UserMsg(name="user", content=user_input)
        )

        # 如果有待确认的工具调用，请求用户确认
        if pending:
            print(f"\n{'-'*40}")
            print(f"Agent 请求确认以下操作:")
            for i, tc in enumerate(pending, 1):
                print(f"  {i}. {tc.name}({tc.input})")

            while True:
                choice = input("\n确认执行? (y/n): ").strip().lower()
                if choice in ("y", "yes", "是", "确认"):
                    # 构建确认结果
                    confirm_results = [
                        ConfirmResult(
                            confirmed=True,
                            tool_call=tc,
                            rules=[
                                PermissionRule(
                                    tool_name=tc.name,
                                    rule_content=None,
                                    behavior=PermissionBehavior.ALLOW,
                                    source="user_confirmed",
                                )
                            ],
                        )
                        for tc in pending
                    ]
                    print(f"\n[用户已确认，Agent 继续执行...]\n")

                    # 发送确认事件，Agent 继续执行
                    async for event in agent.reply_stream(
                        UserConfirmResultEvent(
                            reply_id=agent.state.reply_id,
                            confirm_results=confirm_results,
                        )
                    ):
                        event_type = type(event).__name__
                        if event_type == "TextBlockDeltaEvent":
                            print(event.delta, end="", flush=True)
                        elif event_type == "ToolResultTextDeltaEvent":
                            print(f"  [工具返回] {event.delta}", end="", flush=True)
                        elif event_type == "ReplyEndEvent":
                            print("\n  [回复完成]")
                    break

                elif choice in ("n", "no", "否", "拒绝", "取消"):
                    # 构建拒绝结果
                    confirm_results = [
                        ConfirmResult(
                            confirmed=False,
                            tool_call=tc,
                        )
                        for tc in pending
                    ]
                    print(f"\n[用户已拒绝，Agent 收到错误信息...]\n")

                    # 发送拒绝事件
                    async for event in agent.reply_stream(
                        UserConfirmResultEvent(
                            reply_id=agent.state.reply_id,
                            confirm_results=confirm_results,
                        )
                    ):
                        event_type = type(event).__name__
                        if event_type == "TextBlockDeltaEvent":
                            print(event.delta, end="", flush=True)
                        elif event_type == "ReplyEndEvent":
                            print("\n  [回复完成]")
                    break

                else:
                    print("请输入 y (确认) 或 n (拒绝)")

        print()


if __name__ == "__main__":
    asyncio.run(main())
