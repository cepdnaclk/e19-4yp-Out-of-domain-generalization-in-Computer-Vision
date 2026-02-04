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
from biomedxpro.utils.token_logging import TokenUsageLogger


def _get_api_keys(env_var_name: str) -> list[str]:
    """Helper to fetch rotated keys from environment."""
    raw_keys = os.getenv(env_var_name, "")
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    if not keys:
        logger.warning(f"SECURITY WARNING: No keys found in '{env_var_name}'")
        return []
    return keys


class OpenAIClient(ILLMClient):
    def __init__(
        self, settings: LLMSettings, keys: list[str], token_logger: TokenUsageLogger
    ) -> None:
        self.model_name = settings.model_name
        self.llm_params = settings.llm_params
        self.provider = settings.provider
        self.token_logger = token_logger

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

            # Prompt tokens
            if usage and usage.input_tokens is not None:
                prompt_tokens = usage.input_tokens
                prompt_estimated = False
            else:
                prompt_tokens = len(prompt) // 4
                prompt_estimated = True

            # Completion tokens
            if usage and usage.output_tokens is not None:
                completion_tokens = usage.output_tokens
                completion_estimated = False
            else:
                completion_tokens = len(output_text) // 4
                completion_estimated = True

            # Total tokens
            if usage and usage.total_tokens is not None:
                total_tokens = usage.total_tokens
                total_estimated = False
            else:
                total_tokens = prompt_tokens + completion_tokens
                total_estimated = True

            record = {
                "provider": self.provider,
                "model": self.model_name,
                "prompt_tokens": prompt_tokens,
                "prompt_estimated": prompt_estimated,
                "completion_tokens": completion_tokens,
                "completion_estimated": completion_estimated,
                "total_tokens": total_tokens,
                "total_estimated": total_estimated,
            }

            # Log to TokenUsageLogger
            self.token_logger.log(record)
            return str(response.output_text)

        except OpenAIError as e:
            logger.error(f"LLM Error ({self.model_name}): {e}")
            raise e


class GeminiClient(ILLMClient):
    def __init__(
        self,
        settings: LLMSettings,
        keys: list[str],
        token_logger: TokenUsageLogger,
    ) -> None:
        self.model_name = settings.model_name
        self.llm_params = settings.llm_params
        self.provider = settings.provider
        self.token_logger = token_logger

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

            # Initialize token counts and estimated flags
            if usage:
                # Prompt tokens
                if usage.prompt_token_count is not None:
                    prompt_tokens = usage.prompt_token_count
                    prompt_estimated = False
                else:
                    prompt_tokens = len(prompt) // 4
                    prompt_estimated = True

                # Completion tokens
                if usage.candidates_token_count is not None:
                    completion_tokens = usage.candidates_token_count
                    completion_estimated = False
                else:
                    completion_tokens = len(output_text) // 4
                    completion_estimated = True

                # Total tokens
                if usage.total_token_count is not None:
                    total_tokens = usage.total_token_count
                    total_estimated = False
                else:
                    total_tokens = prompt_tokens + completion_tokens
                    total_estimated = True

            else:
                # Fallback heuristic
                prompt_tokens = len(prompt) // 4
                completion_tokens = len(output_text) // 4
                total_tokens = prompt_tokens + completion_tokens
                prompt_estimated = completion_estimated = total_estimated = True

            record = {
                "provider": self.provider,
                "model": self.model_name,
                "prompt_tokens": prompt_tokens,
                "prompt_estimated": prompt_estimated,
                "completion_tokens": completion_tokens,
                "completion_estimated": completion_estimated,
                "total_tokens": total_tokens,
                "total_estimated": total_estimated,
            }

            self.token_logger.log(record)
            return str(response.text) or ""

        except Exception as e:
            logger.error(f"Gemini Error ({self.model_name}): {e}")
            raise e


def create_llm_client(
    settings: LLMSettings, token_logger: TokenUsageLogger
) -> ILLMClient:
    """
    Factory: Maps provider string to concrete client.
    """
    provider = settings.provider.lower().strip()

    if provider == "openai":
        keys = _get_api_keys("OPENAI_API_KEYS")
        return OpenAIClient(
            settings, keys=keys, token_logger=token_logger
        )  # Uses default OpenAI URL

    elif provider == "groq":
        keys = _get_api_keys("GROQ_API_KEYS")
        return OpenAIClient(settings, keys=keys, token_logger=token_logger)

    elif provider == "gemini":
        keys = _get_api_keys("GEMINI_API_KEYS")
        return GeminiClient(settings, keys=keys, token_logger=token_logger)

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
            return OpenAIClient(settings, keys=keys, token_logger=token_logger)

        raise ValueError(f"Unsupported LLM provider: {provider}")
