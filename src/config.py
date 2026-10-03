from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is an optional convenience
    load_dotenv = None


@dataclass
class LabConfig:
    """Shared configuration for the lab."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "custom": "CUSTOM_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,
    "openrouter": "OPENROUTER_API_KEY",
}


def _api_key_for(provider: str) -> str | None:
    env_name = _API_KEY_ENV.get(provider)
    return os.getenv(env_name) if env_name else None


def _base_url_for(provider: str) -> str | None:
    if provider == "custom":
        return os.getenv("CUSTOM_BASE_URL")
    if provider == "ollama":
        return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    if provider == "openrouter":
        return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    return None


def _provider_config(
    provider_env: str,
    model_env: str,
    temperature_env: str,
    default_provider: str,
    default_model: str,
    default_temperature: float,
) -> ProviderConfig:
    provider = normalize_provider(os.getenv(provider_env, default_provider))
    model_name = os.getenv(model_env, default_model)
    temperature = float(os.getenv(temperature_env, str(default_temperature)))
    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=_api_key_for(provider),
        base_url=_base_url_for(provider),
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a `LabConfig`.

    Supported env vars:
    - `LLM_PROVIDER` / `LLM_MODEL` / `LLM_TEMPERATURE`
    - `JUDGE_PROVIDER` / `JUDGE_MODEL` / `JUDGE_TEMPERATURE`
    - `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`
    - `CUSTOM_BASE_URL` / `CUSTOM_API_KEY`
    - `OLLAMA_BASE_URL`
    - `COMPACT_THRESHOLD_TOKENS` / `COMPACT_KEEP_MESSAGES`
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    if load_dotenv is not None:
        env_path = root / ".env"
        if env_path.exists():
            load_dotenv(env_path)

    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    compact_threshold_tokens = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "600"))
    compact_keep_messages = int(os.getenv("COMPACT_KEEP_MESSAGES", "6"))

    default_provider = os.getenv("LLM_PROVIDER", "openai")

    model = _provider_config(
        provider_env="LLM_PROVIDER",
        model_env="LLM_MODEL",
        temperature_env="LLM_TEMPERATURE",
        default_provider=default_provider,
        default_model="gpt-4o-mini",
        default_temperature=0.2,
    )

    judge_model = _provider_config(
        provider_env="JUDGE_PROVIDER",
        model_env="JUDGE_MODEL",
        temperature_env="JUDGE_TEMPERATURE",
        default_provider=os.getenv("JUDGE_PROVIDER", default_provider),
        default_model=os.getenv("JUDGE_MODEL", "gpt-4o-mini"),
        default_temperature=0.0,
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold_tokens,
        compact_keep_messages=compact_keep_messages,
        model=model,
        judge_model=judge_model,
    )
