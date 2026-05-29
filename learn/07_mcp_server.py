# -*- coding: utf-8 -*-
"""
MCP Server — 用 FastMCP 创建真实的 MCP 服务
==============================================
提供 3 个工具:
  - get_weather: 查询城市天气 (只读)
  - translate_text: 翻译文本 (只读)
  - calculate: 数学计算 (只读)

运行方式:
    .venv\Scripts\python.exe learn\07_mcp_server.py

然后另开一个终端运行 Agent:
    .venv\Scripts\python.exe learn\07_mcp_agent.py
"""

import json
from mcp.server import FastMCP


def create_mcp_server() -> FastMCP:
    """创建并配置 MCP Server"""
    server = FastMCP("my-tools", port=8765)

    # ========================================
    # 工具 1: 天气查询 (只读)
    # ========================================
    @server.tool(
        description="查询指定城市的天气。返回温度、天气状况和湿度。参数: city-城市名称",
        annotations={"readOnlyHint": True},  # 标记为只读工具
    )
    async def get_weather(city: str) -> str:
        weather_data = {
            "北京": {"temp": 25, "condition": "晴", "humidity": 45},
            "上海": {"temp": 28, "condition": "多云", "humidity": 65},
            "广州": {"temp": 32, "condition": "小雨", "humidity": 80},
            "深圳": {"temp": 30, "condition": "阴", "humidity": 70},
            "杭州": {"temp": 26, "condition": "晴", "humidity": 55},
        }
        data = weather_data.get(city)
        if data:
            return f"{city}: {data['condition']}, {data['temp']}°C, 湿度{data['humidity']}%"
        return f"未找到 {city} 的天气数据"

    # ========================================
    # 工具 2: 文本翻译 (只读)
    # ========================================
    @server.tool(
        description="将文本翻译成目标语言。参数: text-要翻译的文本, target_lang-目标语言(zh/en/ja)",
        annotations={"readOnlyHint": True},  # 标记为只读工具
    )
    async def translate_text(text: str, target_lang: str) -> str:
        translations = {
            ("hello", "zh"): "你好",
            ("world", "zh"): "世界",
            ("你好", "en"): "Hello",
            ("世界", "en"): "World",
            ("你好", "ja"): "こんにちは",
            ("人工智能", "en"): "Artificial Intelligence",
            ("人工智能", "ja"): "人工知能",
        }
        key = (text.lower(), target_lang.lower())
        result = translations.get(key)
        if result:
            return f"翻译结果: {text} -> {result}"
        return f"暂无 '{text}' 到 '{target_lang}' 的翻译"

    # ========================================
    # 工具 3: 数学计算 (只读)
    # ========================================
    @server.tool(
        description="执行数学计算。参数: expression-数学表达式, 如 '2**10 + 100'",
        annotations={"readOnlyHint": True},  # 标记为只读工具
    )
    async def calculate(expression: str) -> str:
        try:
            allowed = {"abs": abs, "max": max, "min": min, "round": round}
            result = eval(expression, {"__builtins__": {}}, allowed)
            return f"{expression} = {result}"
        except Exception as e:
            return f"计算错误: {e}"

    return server


def main():
    """启动 MCP Server"""
    print("=" * 60)
    print("[MCP Server] 启动中...")
    print("=" * 60)

    server = create_mcp_server()

    print("\n[已注册工具]")
    print("  1. get_weather(city: str) -> str      查询天气")
    print("  2. translate_text(text, target_lang)   翻译文本")
    print("  3. calculate(expression: str) -> str   数学计算")
    print(f"\n[服务地址] http://127.0.0.1:8765/mcp")
    print("[按 Ctrl+C 停止]\n")

    server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
