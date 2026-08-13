# Pi Agent Python

这是一个参照 [Pi Agent](https://github.com/badlogic/pi-mono) 源码实现的 Python
版本，重点复刻 `packages/agent` 的状态机和 `packages/coding-agent` 的核心编码能力。
它不是简单的“调用一次模型”：模型可以流式回复、连续调用工具、接收工具结果后继续
推理，并把整个过程作为事件流交给 UI 或宿主程序。

参考上游提交：`9795d602306ef68a97585909e8e79f92a389057b`。

## 已实现

- 状态化 `Agent` 与低层 `agent_loop` / `agent_loop_continue`
- `agent_start`、`turn_start`、消息、工具与结束事件
- 串行等待的订阅者；`message_end` 是工具预检前的状态屏障
- steer 与 follow-up 双队列，支持逐条或一次全部取出
- 多工具并行执行；完成事件按完成顺序，结果消息按模型调用顺序
- 工具级串行覆盖、参数校验、前后置钩子和整批 `terminate` 规则
- 模型输出因长度截断时拒绝执行其中的工具调用
- OpenAI-compatible Chat Completions 流式接口
- Pi 风格 `read`、`write`、批量精确 `edit`、`bash` 工具
- `AGENTS.md` / `CLAUDE.md`、`.pi/SYSTEM.md`、`APPEND_SYSTEM.md`
- Agent Skills 元数据发现与提示词注入
- 带 `parentId` 的追加式 JSONL 会话，可从历史消息创建分支
- 交互、print 和 JSONL 事件三种 CLI 使用方式
- 关键实现均附有中文注释

高级终端 TUI、OAuth 登录、Pi 扩展运行时、HTML 导出和上游全部模型供应商并未实现；
模型层通过 `stream_fn` 解耦，可继续添加 Anthropic、Gemini 等适配器。

## 安装

需要 Python 3.11 或更高版本，运行时不依赖第三方包。

```bash
python -m pip install -e .
```

设置兼容 OpenAI Chat Completions 的端点：

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini
pi-agent-py
```

PowerShell：

```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:OPENAI_MODEL = "gpt-4o-mini"
pi-agent-py
```

本地服务可指定地址，API Key 可以留空：

```bash
pi-agent-py --base-url http://127.0.0.1:8000/v1 --model local-model
```

## CLI

```bash
# 交互模式
pi-agent-py

# 单次输出
pi-agent-py -p "检查这个项目并说明如何运行"

# 输出完整 JSONL 事件，适合接入其他程序
pi-agent-py -p --mode json "读取 pyproject.toml"

# 恢复当前项目最近的会话
pi-agent-py --continue

# 全部文件工具默认限制在 cwd 内；显式放开限制
pi-agent-py --allow-outside-cwd
```

`bash` 会执行模型生成的 shell 命令，拥有当前用户权限。只应在可信目录中运行，并按
实际部署环境增加容器、审批钩子或系统级沙箱。

## Python API

```python
import asyncio

from pi_agent import Agent, AgentOptions, Model
from pi_agent.context import build_system_prompt
from pi_agent.providers import openai_compatible_stream
from pi_agent.tools import coding_tools


async def main() -> None:
    cwd = "/path/to/project"
    tools = coding_tools(cwd)
    agent = Agent(
        AgentOptions(
            model=Model(id="gpt-4o-mini", api_key="sk-..."),
            stream_fn=openai_compatible_stream,
            system_prompt=build_system_prompt(cwd, tools),
            tools=tools,
        )
    )

    async def show(event, signal):
        if event["type"] == "message_update":
            update = event["assistantMessageEvent"]
            if update["type"] in {"text_start", "text_delta"}:
                print(update.get("delta", ""), end="", flush=True)

    agent.subscribe(show)
    await agent.prompt("找出测试失败的原因并修复")


asyncio.run(main())
```

工具失败应抛出异常，Agent 会把异常转换为 `isError=true` 的标准工具结果。自定义工具
继承 `AgentTool` 并实现异步 `execute()` 即可。

从已有 `user` 或 `toolResult` 消息尾部重试时，调用 `await agent.continue_()`。
Python 将 `continue` 保留为关键字，因此该方法名比上游 TypeScript API 多一个下划线。

## 事件顺序

无工具调用：

```text
agent_start → turn_start → user message → assistant stream
            → turn_end → agent_end
```

有工具调用：

```text
assistant message
  → tool_execution_start/update/end
  → toolResult message
  → turn_end
  → 下一次 turn_start 与模型请求
```

## 测试

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m compileall -q src tests
```

PowerShell 将前缀写作 `$env:PYTHONPATH="src";`。若已执行 `pip install -e .`，
则无需设置 `PYTHONPATH`。

测试使用确定性的 `scripted_stream`，不会访问网络或消耗模型额度。

## 许可证

MIT。上游归属与参考版本见 [NOTICE.md](NOTICE.md)。
