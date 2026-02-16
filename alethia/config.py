"""Configuration for Alethia, loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class Config:
    openai_api_key: str
    generator_model: str = "gpt-4o"
    verifier_model: str = "gpt-4o"
    reviser_model: str = "gpt-4o"
    max_attempts: int = 5
    execution_timeout: int = 10  # seconds
    max_memory_mb: int = 256
    generator_temperature: float = 0.7
    verifier_temperature: float = 0.2
    reviser_temperature: float = 0.4
    executor_type: str = "local"  # "local" or "judge0"
    judge0_url: str = ""
    judge0_api_key: str = ""
    enable_backtracking: bool = True
    candidates_per_attempt: int = 1
    diverse_generation: bool = True
    llm_provider: str = "openai"  # "openai" or "ollama"
    ollama_base_url: str = "http://localhost:11434/v1"

    def create_openai_client(self) -> OpenAI:
        """Create an OpenAI client configured for the active LLM provider."""
        if self.llm_provider == "ollama":
            return OpenAI(api_key="ollama", base_url=self.ollama_base_url)
        return OpenAI(api_key=self.openai_api_key)

    @classmethod
    def from_env(cls, **overrides) -> Config:
        provider = overrides.pop("llm_provider", None) or os.environ.get("ALETHIA_LLM_PROVIDER", "openai")
        api_key = overrides.pop("openai_api_key", None) or os.environ.get("OPENAI_API_KEY", "")
        if not api_key and provider != "ollama":
            raise ValueError("OPENAI_API_KEY environment variable is required")
        if not api_key:
            api_key = "ollama"
        model = overrides.pop("model", None)
        kwargs: dict = {"openai_api_key": api_key, "llm_provider": provider}
        if model:
            kwargs["generator_model"] = model
            kwargs["verifier_model"] = model
            kwargs["reviser_model"] = model
        env_map: dict[str, tuple[str, type]] = {
            "ALETHIA_MAX_ATTEMPTS": ("max_attempts", int),
            "ALETHIA_EXECUTION_TIMEOUT": ("execution_timeout", int),
            "ALETHIA_MAX_MEMORY_MB": ("max_memory_mb", int),
            "ALETHIA_EXECUTOR": ("executor_type", str),
            "JUDGE0_URL": ("judge0_url", str),
            "JUDGE0_API_KEY": ("judge0_api_key", str),
            "ALETHIA_CANDIDATES_PER_ATTEMPT": ("candidates_per_attempt", int),
            "OLLAMA_BASE_URL": ("ollama_base_url", str),
        }
        for env_var, (field_name, conv) in env_map.items():
            val = os.environ.get(env_var)
            if val is not None:
                kwargs[field_name] = conv(val)
        # ALETHIA_ENABLE_BACKTRACKING: "0" or "false" disables
        bt_val = os.environ.get("ALETHIA_ENABLE_BACKTRACKING")
        if bt_val is not None:
            kwargs.setdefault("enable_backtracking", bt_val.lower() not in ("0", "false", "no"))
        # ALETHIA_DIVERSE_GENERATION: "0" or "false" disables
        dg_val = os.environ.get("ALETHIA_DIVERSE_GENERATION")
        if dg_val is not None:
            kwargs.setdefault("diverse_generation", dg_val.lower() not in ("0", "false", "no"))
        kwargs.update(overrides)
        return cls(**kwargs)
