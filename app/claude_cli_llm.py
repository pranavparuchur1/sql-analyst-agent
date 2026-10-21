"""Run the agent on a Claude subscription through headless Claude Code (`claude -p`).

Set LLM_MODEL=claude-cli:sonnet (or claude-cli:opus / claude-cli:haiku). Calls
count against your Claude plan's usage limits instead of API billing. This is
meant for personal development and evals on your own login; a deployed service
should use an API key (LLM_MODEL=anthropic:...).

The graph only ever calls `llm.with_structured_output(Schema)` and pipes a prompt
into the result (the same small surface tests/fakes.py fakes), so this class
implements just that, rather than a full LangChain BaseChatModel.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

PREFIX = "claude-cli:"


class ClaudeCLIError(RuntimeError):
    pass


@dataclass
class CallStats:
    schema: str
    seconds: float
    api_equiv_usd: float  # what these tokens would cost on the API: NOT a charge on a subscription
    input_tokens: int
    output_tokens: int


def find_claude_binary() -> str:
    """Prefer the native claude.exe over the npm .cmd shim: cmd.exe mangles long quoted arguments."""
    if env := os.environ.get("CLAUDE_BIN"):
        return env
    shim = shutil.which("claude")
    if not shim:
        raise ClaudeCLIError("claude CLI not found. Install Claude Code and run `claude auth login`.")
    native = Path(shim).parent / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
    return str(native) if native.exists() else shim


def render_transcript(messages: list[BaseMessage]) -> tuple[str, str]:
    """Split a chat prompt into (system prompt, user text).

    `claude -p` takes one system prompt and one user turn, so few-shot examples and
    history are rendered as a labelled transcript. Models follow this format well.
    """
    system = "\n\n".join(str(m.content) for m in messages if isinstance(m, SystemMessage))
    turns = [m for m in messages if not isinstance(m, SystemMessage)]
    if len(turns) == 1:
        return system, str(turns[0].content)
    labels = {"human": "User", "ai": "Assistant"}
    body = "\n\n".join(f"### {labels.get(m.type, m.type)}\n{m.content}" for m in turns[:-1])
    return system, (
        f"Conversation so far (examples and earlier turns):\n\n{body}\n\n"
        f"### Current request (answer this one)\n{turns[-1].content}"
    )


class ClaudeCLIModel:
    def __init__(self, model: str = "sonnet", *, binary: str | None = None, timeout_s: float = 300) -> None:
        self.model = model
        self.binary = binary or find_claude_binary()
        self.timeout_s = timeout_s
        # Run from an empty folder so no CLAUDE.md or project settings leak into prompts.
        self.workdir = Path(os.environ.get("CLAUDE_CLI_WORKDIR", Path.home() / ".sql-agent-llm"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.calls: list[CallStats] = []
        self._lock = threading.Lock()  # evals call from several threads

    def command(self, system_prompt: str, json_schema: dict[str, Any]) -> list[str]:
        return [
            self.binary, "-p",
            "--model", self.model,
            # Replacing Claude Code's large default system prompt and disabling tools keeps
            # each call small (less of your usage limit) and stops the CLI acting as an agent.
            "--system-prompt", system_prompt or "You are a precise assistant.",
            "--tools", "",
            "--strict-mcp-config",
            "--setting-sources", "",
            "--no-session-persistence",
            "--output-format", "json",
            "--json-schema", json.dumps(json_schema),
        ]  # fmt: skip

    def invoke_schema(self, schema: type[BaseModel], prompt: PromptValue) -> BaseModel:
        system, user = render_transcript(prompt.to_messages())
        # Strip API credentials so the CLI can only ever bill the signed-in subscription.
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
        started = time.perf_counter()
        proc = subprocess.run(
            self.command(system, schema.model_json_schema()),
            input=user, capture_output=True, text=True, encoding="utf-8",
            env=env, cwd=self.workdir, timeout=self.timeout_s,
        )  # fmt: skip
        try:
            event = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise ClaudeCLIError(f"claude exited {proc.returncode}: {(proc.stderr or proc.stdout)[-500:]}") from e
        if event.get("is_error") or event.get("subtype") != "success":
            raise ClaudeCLIError(f"claude returned an error: {event.get('result') or event.get('subtype')}")
        if event.get("structured_output") is None:
            raise ClaudeCLIError("claude returned no structured_output")

        usage = event.get("usage") or {}
        with self._lock:
            self.calls.append(CallStats(
                schema.__name__, time.perf_counter() - started, float(event.get("total_cost_usd") or 0.0),
                usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
                + usage.get("cache_creation_input_tokens", 0),
                usage.get("output_tokens", 0),
            ))  # fmt: skip
        return schema.model_validate(event["structured_output"])

    def with_structured_output(self, schema: type[BaseModel], **_: Any) -> RunnableLambda:
        # `method=` is accepted and ignored: --json-schema is always native structured output.
        return RunnableLambda(lambda prompt: self.invoke_schema(schema, prompt), name=f"claude-cli:{schema.__name__}")
