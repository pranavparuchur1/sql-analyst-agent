"""Settings from environment variables (and an optional .env file).

Everything that might differ between a laptop, CI and production lives here,
so no other module reads os.environ directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # "provider:model" so switching OpenAI <-> Anthropic is a config change, not a code change.
    llm_model: str = "anthropic:claude-sonnet-5-5"
    # Unset by default: some newer models reject non-default temperatures. Set 0 for
    # older models when you want the most repeatable SQL.
    llm_temperature: float | None = None
    # json_schema = the provider's native structured outputs (constrained decoding).
    # Forced tool calling ('function_calling') isn't supported on some newer models.
    structured_output_method: str = "json_schema"

    db_path: Path = ROOT / "data" / "shop.db"
    checkpoint_db: Path = ROOT / "data" / "checkpoints.db"

    max_attempts: int = 3
    # Queries touching this many tables or more pause for a human; wide joins are
    # where both cost and "plausible but wrong" answers concentrate.
    approval_join_threshold: int = 4
    row_limit: int = 200
    query_timeout_s: float = 5.0
    # Only the most recent messages reach the prompts: enough for follow-ups,
    # bounded so token cost doesn't grow with conversation length.
    history_window: int = 6


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_llm(settings: Settings | None = None, model: str | None = None) -> BaseChatModel:
    """Build the chat model named by LLM_MODEL (or an explicit override, used by evals)."""
    # Imported lazily so tests (which inject a fake LLM) never need a provider package or key.
    from langchain.chat_models import init_chat_model

    settings = settings or get_settings()
    name = model or settings.llm_model
    if name.startswith("claude-cli:"):
        # Subscription mode: headless Claude Code instead of an API key (see app/claude_cli_llm.py).
        from app.claude_cli_llm import ClaudeCLIModel

        return ClaudeCLIModel(name.removeprefix("claude-cli:"))  # type: ignore[return-value]
    kwargs = {} if settings.llm_temperature is None else {"temperature": settings.llm_temperature}
    return init_chat_model(name, **kwargs)
