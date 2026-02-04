# src/biomedxpro/impl/llm_client.py
import os
from itertools import cycle
from typing import Iterator

from google import genai
from google.genai import types
from loguru import logger
from openai import OpenAI, OpenAIError
from openai.types.responses.response import Response

from biomedxpro.core.interfaces import ILLMClient
from biomedxpro.impl.config import LLMSettings
from biomedxpro.utils.usage import record_usage


def _get_api_keys(env_var_name: str) -> list[str]:
    """Helper to fetch rotated keys from environment."""
    raw_keys = os.getenv(env_var_name, "")
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    if not keys:
        logger.warning(f"SECURITY WARNING: No keys found in '{env_var_name}'")
        return []
    return keys


class OpenAIClient(ILLMClient):
    def __init__(self, settings: LLMSettings, keys: list[str]) -> None:
        self.model_name = settings.model_name
        self.llm_params = settings.llm_params
        self.provider = settings.provider

        final_base_url = settings.base_url or None
        if not keys:
            raise ValueError(f"OpenAIClient ({settings.provider}) requires API keys.")

        self._clients = [OpenAI(api_key=k, base_url=final_base_url) for k in keys]
        self._pool: Iterator[OpenAI] = cycle(self._clients)

        logger.info(
            f"Initialized {settings.provider.upper()} client. "
            f"Url: {final_base_url or 'Default'}. Params: {self.llm_params}"
        )

    def generate(self, prompt: str) -> str:
        client = next(self._pool)
        try:
            response: Response = client.responses.create(
                model=self.model_name,
                input=prompt,
                **self.llm_params,
            )

            output_text = response.output_text or ""
            usage = getattr(response, "usage", None)

            # Extract token counts from API response or estimate
            input_tokens = (
                usage.input_tokens
                if usage and usage.input_tokens is not None
                else len(prompt) // 4
            )
            output_tokens = (
                usage.output_tokens
                if usage and usage.output_tokens is not None
                else len(output_text) // 4
            )
            # Reasoning tokens are provider-specific (e.g., o1 models)
            reasoning_tokens = (
                getattr(usage, "reasoning_tokens", 0) if usage else 0
            ) or 0

            record_usage(
                provider=self.provider,
                model=self.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
            )
            return str(response.output_text)

        except OpenAIError as e:
            logger.error(f"LLM Error ({self.model_name}): {e}")
            raise e


class GeminiClient(ILLMClient):
    def __init__(
        self,
        settings: LLMSettings,
        keys: list[str],
    ) -> None:
        self.model_name = settings.model_name
        self.llm_params = settings.llm_params
        self.provider = settings.provider

        if not keys:
            raise ValueError("GeminiClient requires at least one API key.")

        self._clients = [genai.Client(api_key=k) for k in keys]
        self._pool: Iterator[genai.Client] = cycle(self._clients)

        logger.info(
            f"Initialized Gemini client. Model: {self.model_name}. Params: {self.llm_params}"
        )

    def generate(self, prompt: str) -> str:
        client = next(self._pool)
        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents={"text": prompt},
                config=types.GenerateContentConfig(**self.llm_params),
            )

            output_text = response.text or ""
            usage = getattr(response, "usage_metadata", None)

            # Extract token counts from API response or estimate
            input_tokens = (
                usage.prompt_token_count
                if usage and usage.prompt_token_count is not None
                else len(prompt) // 4
            )
            output_tokens = (
                usage.candidates_token_count
                if usage and usage.candidates_token_count is not None
                else len(output_text) // 4
            )
            # Gemini doesn't typically report reasoning tokens separately
            reasoning_tokens = 0

            record_usage(
                provider=self.provider,
                model=self.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
            )
            return str(response.text) or ""

        except Exception as e:
            logger.error(f"Gemini Error ({self.model_name}): {e}")
            raise e


def create_llm_client(settings: LLMSettings) -> ILLMClient:
    """
    Factory: Maps provider string to concrete client.
    """
    provider = settings.provider.lower().strip()

    if provider == "openai":
        keys = _get_api_keys("OPENAI_API_KEYS")
        return OpenAIClient(settings, keys=keys)  # Uses default OpenAI URL

    elif provider == "groq":
        keys = _get_api_keys("GROQ_API_KEYS")
        return OpenAIClient(settings, keys=keys)

    elif provider == "gemini":
        keys = _get_api_keys("GEMINI_API_KEYS")
        return GeminiClient(settings, keys=keys)

    else:
        # Fallback: If unknown provider (e.g. "deepseek"), try generic OpenAI client
        # This assumes the researcher provided a base_url in the settings
        if settings.base_url:
            logger.info(
                f"Provider '{provider}' unknown, attempting generic OpenAI connection to {settings.base_url}"
            )
            keys = _get_api_keys(f"{provider.upper()}_API_KEYS")
            if not keys:
                raise ValueError(
                    f"No API keys found for provider '{provider}' in environment variable '{provider.upper()}_API_KEYS'"
                )
            return OpenAIClient(settings, keys=keys)

        raise ValueError(f"Unsupported LLM provider: {provider}")
