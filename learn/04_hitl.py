# -*- coding: utf-8 -*-
"""
实战 4: Human-in-the-Loop (HITL)
=================================
场景: Agent 在执行危险操作前请求用户确认

流程:
  1. Agent 尝试调用危险工具 (如删除文件)
  2. 权限引擎检查 -> 返回 ASK (需要用户确认)
  3. Agent 产出 RequireUserConfirmEvent，暂停执行
  4. 用户确认 -> UserConfirmResultEvent
  5. Agent 继续执行

运行方式:
    D:\code_project\github-proj\agentscope\.venv\Scripts\python.exe learn\04_hitl.py
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
from agentscope.message import TextBlock, UserMsg, ToolCallBlock
from agentscope.event import (
    RequireUserConfirmEvent,
    UserConfirmResultEvent,
    ConfirmResult,
)


# ============================================================
# 工具 1: 安全的计算器 (总是允许)
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
            message="计算器是安全的，允许执行",
        )

    async def __call__(self, expression: str, **kwargs) -> ToolChunk:
        try:
            allowed = {"abs": abs, "max": max, "min": min, "round": round}
            result = eval(expression, {"__builtins__": {}}, allowed)
            return ToolChunk(content=[TextBlock(text=f"结果: {result}")])
        except Exception as e:
            return ToolChunk(content=[TextBlock(text=f"错误: {e}")])


# ============================================================
# 工具 2: 文件删除 (危险操作，需要用户确认)
# ============================================================
class DeleteFileTool(ToolBase):
    """
    模拟文件删除工具 - 危险操作，需要用户确认

    关键: check_permissions 返回 ASK，触发 Human-in-the-Loop
    """

    name = "delete_file"
    description = (
        "【危险操作】删除指定路径的文件。此操作不可撤销！"
        "调用此工具会触发用户确认流程。"
        "参数: file_path - 要删除的文件路径"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "要删除的文件路径, 如 '/tmp/test.txt'",
            }
        },
        "required": ["file_path"],
    }
    is_concurrency_safe = False
    is_read_only = False  # 写操作

    _deleted_files: list[str] = []

    async def check_permissions(self, tool_input, context):
        """
        返回 ASK 触发 Human-in-the-Loop 确认流程
        """
        file_path = tool_input.get("file_path", "")
        return PermissionDecision(
            behavior=PermissionBehavior.ASK,
            message=f"危险操作确认: 删除文件 '{file_path}'",
            suggested_rules=[
                PermissionRule(
                    tool_name="delete_file",
                    rule_content=None,
                    behavior=PermissionBehavior.ALLOW,
                    source="suggested",
                )
            ],
        )

    async def __call__(self, file_path: str, **kwargs) -> ToolChunk:
        self._deleted_files.append(file_path)
        return ToolChunk(content=[
            TextBlock(text=f"文件 '{file_path}' 已成功删除")
        ])


# ============================================================
# 辅助函数: 流式输出 + 捕获 HITL 事件
# ============================================================
async def run_agent_and_capture(
    agent: Agent,
    user_msg: UserMsg,
) -> tuple[str, list[RequireUserConfirmEvent]]:
    """
    运行 Agent，捕获 RequireUserConfirmEvent

    返回: (回复文本, 待确认事件列表)
    """
    full_text = ""
    pending_confirms: list[RequireUserConfirmEvent] = []
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

        elif event_type == "RequireUserConfirmEvent":
            # 捕获 HITL 事件！
            pending_confirms.append(event)
            print(f"\n  [!] 需要用户确认: {len(event.tool_calls)} 个工具调用")
            for tc in event.tool_calls:
                print(f"      - {tc.name}: {tc.input}")

        elif event_type == "ReplyEndEvent":
            print("\n")

    return full_text, pending_confirms


# ============================================================
# 主程序: Human-in-the-Loop 演示
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
    toolkit = Toolkit(tools=[CalculatorTool(), DeleteFileTool()])

    # 3. 创建 Agent
    agent = Agent(
        name="HITL-Demo",
        system_prompt=(
            "你是一个 AI 助手。你可以使用以下工具:\n"
            "- calculator: 执行数学计算 (安全工具，直接执行)\n"
            "- delete_file: 删除文件 (危险工具，会触发用户确认)\n"
            "\n"
            "重要规则:\n"
            "1. 当用户要求删除文件时，你必须调用 delete_file 工具\n"
            "2. 不要只用文字描述，必须实际调用工具\n"
            "3. delete_file 会触发用户确认流程，这是正常的\n"
            "4. 用中文回答。"
        ),
        model=model,
        toolkit=toolkit,
        react_config=ReActConfig(max_iters=10),
    )

    # ========================================
    # 场景 1: 安全操作 (直接执行)
    # ========================================
    print("=" * 60)
    print("[场景 1] 安全操作 - 计算器")
    print("=" * 60)

    text, confirms = await run_agent_and_capture(
        agent, UserMsg(name="user", content="计算 2 的 10 次方")
    )
    print(f"\n[结果] 待确认: {len(confirms)} (预期: 0)")

    # ========================================
    # 场景 2: 危险操作 (触发 HITL -> 用户确认)
    # ========================================
    print("\n" + "=" * 60)
    print("[场景 2] 危险操作 - 删除文件 (触发 HITL)")
    print("=" * 60)

    text, confirms = await run_agent_and_capture(
        agent,
        UserMsg(name="user", content="请删除临时文件 /tmp/test.txt"),
    )
    print(f"\n[结果] 待确认: {len(confirms)} (预期: 1)")

    if confirms:
        # 用户确认
        confirm_event = confirms[0]
        print(f"\n[用户] 确认执行:")
        for tc in confirm_event.tool_calls:
            print(f"  - {tc.name}({tc.input})")

        confirm_results = [
            ConfirmResult(
                confirmed=True,
                tool_call=tc,
                rules=[
                    PermissionRule(
                        tool_name=tc.name,
                        rule_content=None,
                        behavior=PermissionBehavior.ALLOW,
                        source="user",
                    )
                ],
            )
            for tc in confirm_event.tool_calls
        ]

        result_event = UserConfirmResultEvent(
            reply_id=confirm_event.reply_id,
            confirm_results=confirm_results,
        )

        print("\n[Agent 继续执行...]")
        async for event in agent.reply_stream(result_event):
            et = type(event).__name__
            if et == "TextBlockDeltaEvent":
                print(event.delta, end="", flush=True)
            elif et == "ToolResultTextDeltaEvent":
                print(f"  [工具返回]: {event.delta}", end="", flush=True)
            elif et == "ReplyEndEvent":
                print("\n[回复完成]")
    else:
        print("\n[!] 未触发 HITL，Agent 可能没有调用 delete_file 工具")

    # ========================================
    # 总结
    # ========================================
    print("\n" + "=" * 60)
    print("[HITL 流程总结]")
    print("=" * 60)
    print("""
Human-in-the-Loop 完整流程:

  1. 用户: "删除 /tmp/test.txt"
  2. Agent 推理 -> 决定调用 delete_file 工具
  3. 权限引擎检查:
     - DeleteFileTool.check_permissions() 返回 ASK
     - 引擎返回 RequireUserConfirmEvent
  4. Agent 暂停，等待用户确认
  5. 用户确认 -> UserConfirmResultEvent(confirmed=True)
  6. Agent 继续执行 delete_file 工具
  7. 返回最终结果

  关键 API:
  - PermissionDecision(behavior=ASK)  : 触发 HITL
  - RequireUserConfirmEvent          : Agent 请求确认
  - UserConfirmResultEvent           : 用户回复
  - ConfirmResult(confirmed=True/False): 确认/拒绝
    """)


if __name__ == "__main__":
    asyncio.run(main())
