"""Offline tests for subscription mode: subprocess is mocked, so nothing is called or billed."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.claude_cli_llm import ClaudeCLIError, ClaudeCLIModel, render_transcript
from app.config import Settings, get_llm
from app.prompts import ROUTER_PROMPT, SQL_PROMPT
from app.schemas import IntentDecision


@pytest.fixture()
def model(tmp_path, monkeypatch) -> ClaudeCLIModel:
    monkeypatch.setenv("CLAUDE_CLI_WORKDIR", str(tmp_path))
    return ClaudeCLIModel("sonnet", binary="claude-fake")


def fake_run(event: dict | None, captured: dict, stdout: str | None = None):
    def run(cmd, **kwargs):
        captured.update(cmd=cmd, **kwargs)
        return SimpleNamespace(returncode=0, stdout=stdout if stdout is not None else json.dumps(event), stderr="")

    return run


def router_prompt():
    return ROUTER_PROMPT.invoke({"history": [], "question": "Top 5 products?"})


def test_structured_output_round_trip_and_safety_flags(model, monkeypatch):
    captured: dict = {}
    event = {"type": "result", "subtype": "success", "is_error": False, "total_cost_usd": 0.004,
             "usage": {"input_tokens": 5, "cache_read_input_tokens": 700, "output_tokens": 40},
             "structured_output": {"intent": "data_question", "reason": "asks for data"}}  # fmt: skip
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setattr(subprocess, "run", fake_run(event, captured))

    out = (ROUTER_PROMPT | model.with_structured_output(IntentDecision, method="json_schema")).invoke(
        {"history": [], "question": "Top 5 products?"}
    )

    assert out == IntentDecision(intent="data_question", reason="asks for data")
    cmd = captured["cmd"]
    assert cmd[cmd.index("--tools") + 1] == ""  # no file/shell tools for the CLI agent
    assert json.loads(cmd[cmd.index("--json-schema") + 1])["title"] == "IntentDecision"
    assert "front door of a data assistant" in cmd[cmd.index("--system-prompt") + 1]
    assert captured["input"] == "Top 5 products?"
    assert "ANTHROPIC_API_KEY" not in captured["env"]  # can only bill the subscription
    assert model.calls[0].input_tokens == 705 and model.calls[0].api_equiv_usd == 0.004


@pytest.mark.parametrize(
    ("event", "stdout", "match"),
    [
        ({"subtype": "error_max_turns", "is_error": True, "result": "limit"}, None, "returned an error"),
        ({"subtype": "success", "is_error": False, "structured_output": None}, None, "no structured_output"),
        (None, "Not logged in. Please run /login", "exited"),
    ],
)
def test_errors_are_reported(model, monkeypatch, event, stdout, match):
    monkeypatch.setattr(subprocess, "run", fake_run(event, {}, stdout))
    with pytest.raises(ClaudeCLIError, match=match):
        model.invoke_schema(IntentDecision, router_prompt())


def test_transcript_renders_few_shot_and_history():
    prompt = SQL_PROMPT.invoke(
        {"dialect": "sqlite", "row_limit": 10, "rules": "R", "schema": "S", "retry_feedback": "",
         "history": [HumanMessage("Top products?"), AIMessage("4K Monitor\n\n[SQL used]\nSELECT 1")],
         "question": "now only for Texas"}
    )  # fmt: skip
    system, user = render_transcript(prompt.to_messages())
    assert "Schema:\nS" in system
    assert "### User\nHow many orders were delivered in March 2026?" in user  # few-shot example
    assert "[SQL used]\nSELECT 1" in user  # history carries the previous SQL
    assert user.rstrip().endswith("### Current request (answer this one)\nnow only for Texas")


def test_get_llm_selects_cli_mode(monkeypatch):
    monkeypatch.setenv("CLAUDE_BIN", "claude-fake")
    llm = get_llm(Settings(_env_file=None), "claude-cli:haiku")
    assert isinstance(llm, ClaudeCLIModel) and llm.model == "haiku"
