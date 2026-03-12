"""LLM provider factory supporting OpenAI, Anthropic, and local models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel

if TYPE_CHECKING:
    from diffusion_agent.config import Settings


def create_llm(config: Settings) -> BaseChatModel:
    """Create an LLM instance based on the configured provider."""
    if config.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.llm_model,
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
        )
    elif config.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.llm_model,
            api_key=config.llm_api_key,
        )
    elif config.llm_provider == "local":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.llm_model,
            api_key=config.llm_api_key or "not-needed",
            base_url=config.llm_base_url or "http://localhost:8000/v1",
        )
    else:
        raise ValueError(f"Unknown LLM provider: {config.llm_provider}")
