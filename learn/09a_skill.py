# -*- coding: utf-8 -*-
"""
实战 9a: Skill — 可复用的 Agent 能力包
========================================
Skill 是一个包含 SKILL.md 的目录，描述了 Agent 应该如何完成特定任务。
AgentScope 会从目录加载 Skill，自动注入到 Agent 的 system_prompt 中。

运行方式:
    .venv\Scripts\python.exe learn\09a_skill.py
"""

import os
import sys
import asyncio
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
from agentscope.skill import LocalSkillLoader


# ============================================================
# 第一步: 创建 Skill 目录
# ============================================================
def create_skill_directory():
    """创建一个 Skill 示例目录"""
    skill_dir = Path(__file__).parent / "skills" / "code-reviewer"
    skill_dir.mkdir(parents=True, exist_ok=True)

    # SKILL.md — Skill 的核心文件 (frontmatter + markdown)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("""---
name: Code Reviewer
description: 专业代码审查技能，帮助审查 Python 代码的质量、安全性和性能
---

# 代码审查技能

你是一个专业的 Python 代码审查员。请按以下维度审查代码:

## 审查维度

### 1. 代码风格
- 命名规范（变量、函数、类）
- PEP 8 格式
- 注释完整性

### 2. 功能正确性
- 逻辑是否正确
- 边界条件处理
- 错误处理机制

### 3. 安全性
- SQL 注入风险
- XSS 漏洞
- 敏感信息泄露

### 4. 性能
- 时间复杂度
- 空间复杂度
- 不必要的重复计算

## 输出格式

请按以下格式输出审查报告:

```
## 审查报告

### 发现的问题
1. [严重/警告/建议] 问题描述

### 修复建议
- 具体建议

### 总结
[总体评价]
```
""", encoding="utf-8")

    # 示例代码文件
    example_py = skill_dir / "example.py"
    example_py.write_text("""# 待审查的示例代码

def calculate_average(numbers):
    if len(numbers) == 0:
        return 0
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)


def get_user_data(user_id):
    # TODO: 从数据库获取用户数据
    query = "SELECT * FROM users WHERE id = " + user_id
    return query


class UserManager:
    def __init__(self):
        self.users = []

    def add_user(self, user):
        self.users.append(user)

    def get_user(self, index):
        return self.users[index]
""", encoding="utf-8")

    print(f"[Skill] 已创建 Skill 目录: {skill_dir}")
    print(f"  - SKILL.md: Skill 说明文件")
    print(f"  - example.py: 示例代码")
    return skill_dir


# ============================================================
# 第二步: 加载 Skill
# ============================================================
async def main():
    print("=" * 60)
    print("Skill — 可复用的 Agent 能力包")
    print("=" * 60)

    # 1. 创建 Skill 目录
    skill_dir = create_skill_directory()

    # 2. 使用 LocalSkillLoader 加载 Skill (scan_subdir=True 扫描子目录)
    loader = LocalSkillLoader(str(skill_dir.parent), scan_subdir=True)
    skills = await loader.list_skills()

    print(f"\n[Skill] 可用 Skills: {len(skills)} 个")
    for skill in skills:
        print(f"  - {skill.name}: {skill.description}")
        print(f"    目录: {skill.dir}")

    # 3. 创建带有 Skill 的 Agent
    if skills:
        skill = skills[0]
        print(f"\n[Skill] 加载 Skill: {skill.name}")
        print(f"  内容预览:\n{skill.markdown[:150]}...")

        agent = Agent(
            name="Code-Reviewer",
            system_prompt=(
                "你是一个专业的代码审查员。\n\n"
                f"## 技能说明\n{skill.markdown}\n\n"
                "请按照上述技能要求审查代码，用中文回答。"
            ),
            model=OpenAIChatModel(
                credential=OpenAICredential(
                    api_key=os.environ["OPENAI_API_KEY"],
                    base_url=os.environ.get("OPENAI_BASE_URL"),
                ),
                model="LongCat-2.0-Preview",
                stream=False,
                context_size=128000,
            ),
            toolkit=Toolkit(tools=[]),
            react_config=ReActConfig(max_iters=5),
        )

        # 4. 使用 Agent 审查代码
        example_code = (Path(skill.dir) / "example.py").read_text(encoding="utf-8")
        print(f"\n[Skill] 使用 Agent 审查代码...")
        result = await agent.reply(
            UserMsg(name="user", content=f"请审查以下代码:\n\n```python\n{example_code}\n```")
        )
        print(f"\n[审查结果]\n{result.content}")

    # ========================================
    # 总结
    # ========================================
    print("\n" + "=" * 60)
    print("[Skill 总结]")
    print("=" * 60)
    print("""
Skill 核心概念:
  1. Skill 是一个包含 SKILL.md 的目录
  2. SKILL.md 使用 frontmatter (YAML) 定义 name 和 description
  3. LocalSkillLoader 从目录加载所有 Skill
  4. Skill 内容注入到 Agent 的 system_prompt 中
  5. Agent 按照 Skill 的指导执行任务

Skill 目录结构:
  skills/
  └── code-reviewer/
      ├── SKILL.md      ← 技能说明 (必须)
      ├── example.py    ← 示例文件 (可选)
      └── templates/    ← 模板文件 (可选)

使用场景:
  - 代码审查 (Code Review)
  - 数据分析 (Data Analysis)
  - 文档写作 (Writing)
  - 任何需要标准化流程的任务
    """)


if __name__ == "__main__":
    asyncio.run(main())
