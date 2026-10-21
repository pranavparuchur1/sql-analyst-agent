"""A scripted stand-in for a chat model, so graph tests run offline, free and deterministic.

The graph only ever calls `llm.with_structured_output(Schema)` and pipes a prompt
into the result, so that is the whole surface we need to fake.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel


class ScriptedLLM:
    def __init__(self, script: dict[type[BaseModel], list[Any]] | None = None) -> None:
        # Responses are queued per schema, so a test reads like the conversation it simulates.
        self.script: dict[type[BaseModel], list[Any]] = {k: list(v) for k, v in (script or {}).items()}
        self.prompts: dict[str, list[PromptValue]] = defaultdict(list)
        self.structured_kwargs: dict[str, dict[str, Any]] = {}

    def add(self, schema: type[BaseModel], *responses: Any) -> ScriptedLLM:
        self.script.setdefault(schema, []).extend(responses)
        return self

    def with_structured_output(self, schema: type[BaseModel], **kwargs: Any) -> RunnableLambda:
        self.structured_kwargs[schema.__name__] = kwargs

        def respond(prompt: PromptValue) -> BaseModel:
            self.prompts[schema.__name__].append(prompt)
            queue = self.script.get(schema)
            if not queue:
                raise AssertionError(f"ScriptedLLM has no response left for {schema.__name__}")
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item if isinstance(item, schema) else schema(**item)

        return RunnableLambda(respond)

    def prompt_text(self, schema_name: str, index: int = -1) -> str:
        """The full text of a recorded prompt, for asserting what the model was shown."""
        return "\n".join(str(m.content) for m in self.prompts[schema_name][index].to_messages())

    def remaining(self) -> dict[str, int]:
        return {k.__name__: len(v) for k, v in self.script.items() if v}
