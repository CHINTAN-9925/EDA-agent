"""Resilient Groq/OpenRouter client using OpenAI-compatible Chat APIs."""

from __future__ import annotations

import json
import os
import re
import time
from typing import TypeVar

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, OpenAI, RateLimitError
from pydantic import BaseModel, ValidationError

load_dotenv()

T = TypeVar("T", bound=BaseModel)


class LLMServiceError(RuntimeError):
    """A user-readable LLM provider failure."""


class LLMClient:
    """Small provider-neutral wrapper with bounded retry and Pydantic parsing."""

    def __init__(self, api_key: str | None = None, model: str | None = None, provider: str | None = None, timeout: float = 60.0, retries: int = 1) -> None:
        self.provider = (provider or os.getenv("LLM_PROVIDER", "groq")).strip().lower()
        if self.provider == "groq":
            key_variable = "GROQ_API_KEY"
            self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
            base_url = "https://api.groq.com/openai/v1"
            headers = None
        elif self.provider == "openrouter":
            key_variable = "OPENROUTER_API_KEY"
            self.model = model or os.getenv("OPENROUTER_MODEL", "openrouter/free")
            base_url = "https://openrouter.ai/api/v1"
            headers = {"X-OpenRouter-Title": "Agentic AI Data Analyst"}
        else:
            raise LLMServiceError("LLM_PROVIDER must be 'groq' or 'openrouter'.")
        self.provider_name = "Groq" if self.provider == "groq" else "OpenRouter"
        self.api_key = api_key or os.getenv(key_variable, "")
        self.retries = max(0, min(retries, 2))
        if not self.api_key:
            raise LLMServiceError(f"{key_variable} is missing. Add it to .env before running agentic EDA.")
        self.client = OpenAI(
            base_url=base_url,
            api_key=self.api_key,
            timeout=timeout,
            max_retries=0,
            default_headers=headers,
        )

    def _request(self, system: str, user: str, json_mode: bool) -> str:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.1,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if not content:
            raise LLMServiceError(f"{self.provider_name} returned an empty response.")
        return content

    def complete_text(self, system: str, user: str) -> str:
        """Request normal text with a small retry budget."""
        return self._with_retries(system, user, json_mode=False)

    def complete_structured(self, system: str, user: str, schema: type[T]) -> T:
        """Request JSON and validate it; retry once with validation feedback."""
        last_error: Exception | None = None
        prompt = user
        for attempt in range(self.retries + 1):
            try:
                content = self._request(system, prompt, json_mode=(attempt == 0))
                data = self._extract_json(content)
                return schema.model_validate(data)
            except (json.JSONDecodeError, ValidationError, LLMServiceError) as exc:
                last_error = exc
                prompt = user + f"\nPrevious response was invalid ({exc}). Return one valid JSON object only."
            except APIStatusError as exc:
                # Some free models reject response_format; retry without it.
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.5)
                    continue
                raise self._friendly_error(exc) from exc
        raise LLMServiceError(f"The model did not return valid structured JSON: {last_error}")

    def _with_retries(self, system: str, user: str, json_mode: bool) -> str:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return self._request(system, user, json_mode)
            except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.75 * (attempt + 1))
                    continue
                raise self._friendly_error(exc) from exc
        raise LLMServiceError(str(last_error))

    @staticmethod
    def _extract_json(content: str) -> dict:
        text = content.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        elif not text.startswith("{"):
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                text = text[start:end + 1]
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise json.JSONDecodeError("Expected a JSON object", text, 0)
        return parsed

    def _friendly_error(self, exc: Exception) -> LLMServiceError:
        if isinstance(exc, AuthenticationError):
            return LLMServiceError(f"{self.provider_name} rejected the API key. Check the configured credential.")
        if isinstance(exc, RateLimitError):
            return LLMServiceError(f"{self.provider_name} rate limit reached. Wait briefly or choose another model.")
        if isinstance(exc, APITimeoutError):
            return LLMServiceError(f"{self.provider_name} timed out. Retry or choose a faster model.")
        if isinstance(exc, APIConnectionError):
            return LLMServiceError(f"Could not connect to {self.provider_name}. Check the network and retry.")
        status = getattr(exc, "status_code", None)
        if status == 404:
            return LLMServiceError(f"The configured {self.provider_name} model is unavailable. Choose a current model ID.")
        return LLMServiceError(f"{self.provider_name} request failed: {exc}")


class OpenRouterLLM(LLMClient):
    """Backward-compatible client that explicitly selects OpenRouter."""

    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: float = 60.0, retries: int = 1) -> None:
        super().__init__(api_key=api_key, model=model, provider="openrouter", timeout=timeout, retries=retries)
