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
            return str(response.output_text)
        except OpenAIError as e:
            logger.error(f"LLM Error ({self.model_name}): {e}")
            raise e


class GeminiClient(ILLMClient):
    def __init__(self, settings: LLMSettings, keys: list[str]) -> None:
        self.model_name = settings.model_name
        self.llm_params = settings.llm_params

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
            return response.text or ""
        except Exception as e:
            logger.error(f"Gemini Error: {e}")
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
        return OpenAIClient(
            settings,
            keys=keys,
        )

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
