"""Ollama client wrapper for MCP Course integration.

This module provides a wrapper around the Ollama client with course-specific
functionality, error handling, and privacy-by-design features.
"""

from collections.abc import AsyncGenerator
import logging
from typing import Any

import httpx
import ollama

from mcp_course.ollama_client.config import OllamaConfig


logger = logging.getLogger(__name__)


class OllamaConnectionError(Exception):
    """Raised when connection to Ollama server fails."""
    pass


class OllamaModelError(Exception):
    """Raised when model-related operations fail."""
    pass


class OllamaClient:
    """Wrapper for Ollama client with course-specific functionality.

    This client provides a simplified interface for interacting with Ollama
    while ensuring privacy compliance and proper error handling.
    """

    def __init__(self, config: OllamaConfig):
        """Initialize the Ollama client.

        Args:
            config: Ollama configuration instance
        """
        self.config = config
        self._sync_client: ollama.Client | None = None
        self._async_client: ollama.AsyncClient | None = None

        # Validate configuration
        config.validate()

        logger.info(f"Initialized Ollama client for model: {config.model_name}")

    @property
    def sync_client(self) -> ollama.Client:
        """Get synchronous Ollama client."""
        if self._sync_client is None:
            self._sync_client = ollama.Client(
                host=self.config.endpoint,
                timeout=self.config.timeout
            )
        return self._sync_client

    @property
    def async_client(self) -> ollama.AsyncClient:
        """Get asynchronous Ollama client."""
        if self._async_client is None:
            self._async_client = ollama.AsyncClient(
                host=self.config.endpoint,
                timeout=self.config.timeout
            )
        return self._async_client

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        stream: bool = False,
        **kwargs
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Generate response from Ollama model.

        Args:
            prompt: User prompt
            system: Optional system message
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            Response dictionary or async generator for streaming

        Raises:
            OllamaConnectionError: If connection fails
            OllamaModelError: If model operation fails
        """
        try:
            # Merge parameters with config defaults
            params = {**self.config.parameters, **kwargs}

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = self.sync_client.chat(
                model=self.config.model_name,
                messages=messages,
                stream=stream,
                options=params
            )

            return response

        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Failed to connect to Ollama server: {e}") from e
        except Exception as e:
            raise OllamaModelError(f"Model generation failed: {e}") from e

    async def generate_async(
        self,
        prompt: str,
        system: str | None = None,
        stream: bool = False,
        **kwargs
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Generate response from Ollama model asynchronously.

        Args:
            prompt: User prompt
            system: Optional system message
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            Response dictionary or async generator for streaming

        Raises:
            OllamaConnectionError: If connection fails
            OllamaModelError: If model operation fails
        """
        try:
            # Merge parameters with config defaults
            params = {**self.config.parameters, **kwargs}

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = await self.async_client.chat(
                model=self.config.model_name,
                messages=messages,
                stream=stream,
                options=params
            )

            return response

        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Failed to connect to Ollama server: {e}") from e
        except Exception as e:
            raise OllamaModelError(f"Model generation failed: {e}") from e

    def list_models(self) -> list[dict[str, Any]]:
        """List available models on Ollama server.

        Returns:
            List of model information dictionaries

        Raises:
            OllamaConnectionError: If connection fails
        """
        try:
            response = self.sync_client.list()
            return response.get('models', [])
        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Failed to connect to Ollama server: {e}") from e
        except Exception as e:
            raise OllamaModelError(f"Failed to list models: {e}") from e

    async def list_models_async(self) -> list[dict[str, Any]]:
        """List available models on Ollama server asynchronously.

        Returns:
            List of model information dictionaries

        Raises:
            OllamaConnectionError: If connection fails
        """
        try:
            response = await self.async_client.list()
            return response.get('models', [])
        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Failed to connect to Ollama server: {e}") from e
        except Exception as e:
            raise OllamaModelError(f"Failed to list models: {e}") from e

    def is_model_available(self, model_name: str | None = None) -> bool:
        """Check if a model is available on the server.

        Args:
            model_name: Model name to check (defaults to config model)

        Returns:
            True if model is available
        """
        target_model = model_name or self.config.model_name

        try:
            models = self.list_models()
            available_models = []
            for model in models:
                model_name = model.get('model')
                available_models.append(model_name)

            return target_model in available_models
        except (OllamaConnectionError, OllamaModelError):
            return False

    async def is_model_available_async(self, model_name: str | None = None) -> bool:
        """Check if a model is available on the server asynchronously.

        Args:
            model_name: Model name to check (defaults to config model)

        Returns:
            True if model is available
        """
        target_model = model_name or self.config.model_name

        try:
            models = await self.list_models_async()
            available_models = []
            for model in models:
                if isinstance(model, dict):
                    # Try different possible key names
                    model_name = model.get('name') or model.get('model') or model.get('id')
                    if model_name:
                        available_models.append(model_name)
                elif hasattr(model, 'name'):
                    available_models.append(model.name)
                else:
                    # Fallback: convert to string and hope it's the model name
                    available_models.append(str(model))

            return target_model in available_models
        except (OllamaConnectionError, OllamaModelError):
            return False

    def pull_model(self, model_name: str | None = None) -> bool:
        """Pull a model to the Ollama server.

        Args:
            model_name: Model name to pull (defaults to config model)

        Returns:
            True if model was pulled successfully

        Raises:
            OllamaConnectionError: If connection fails
            OllamaModelError: If model pull fails
        """
        target_model = model_name or self.config.model_name

        try:
            logger.info(f"Pulling model: {target_model}")
            self.sync_client.pull(target_model)
            logger.info(f"Successfully pulled model: {target_model}")
            return True
        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Failed to connect to Ollama server: {e}") from e
        except Exception as e:
            raise OllamaModelError(f"Failed to pull model {target_model}: {e}") from e

    async def pull_model_async(self, model_name: str | None = None) -> bool:
        """Pull a model to the Ollama server asynchronously.

        Args:
            model_name: Model name to pull (defaults to config model)

        Returns:
            True if model was pulled successfully

        Raises:
            OllamaConnectionError: If connection fails
            OllamaModelError: If model pull fails
        """
        target_model = model_name or self.config.model_name

        try:
            logger.info(f"Pulling model: {target_model}")
            await self.async_client.pull(target_model)
            logger.info(f"Successfully pulled model: {target_model}")
            return True
        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Failed to connect to Ollama server: {e}") from e
        except Exception as e:
            raise OllamaModelError(f"Failed to pull model {target_model}: {e}") from e

    def get_model_info(self, model_name: str | None = None) -> dict[str, Any] | None:
        """Get information about a specific model.

        Args:
            model_name: Model name to get info for (defaults to config model)

        Returns:
            Model information dictionary or None if not found
        """
        target_model = model_name or self.config.model_name

        try:
            models = self.list_models()
            for model in models:
                if isinstance(model, dict):
                    # Try different possible key names
                    model_name = model.get('name') or model.get('model') or model.get('id')
                    if model_name == target_model:
                        return model
                elif hasattr(model, 'name') and model.name == target_model:
                    return model
            return None
        except (OllamaConnectionError, OllamaModelError):
            return None

    async def get_model_info_async(self, model_name: str | None = None) -> dict[str, Any] | None:
        """Get information about a specific model asynchronously.

        Args:
            model_name: Model name to get info for (defaults to config model)

        Returns:
            Model information dictionary or None if not found
        """
        target_model = model_name or self.config.model_name

        try:
            models = await self.list_models_async()
            for model in models:
                if isinstance(model, dict):
                    # Try different possible key names
                    model_name = model.get('name') or model.get('model') or model.get('id')
                    if model_name == target_model:
                        return model
                elif hasattr(model, 'name') and model.name == target_model:
                    return model
            return None
        except (OllamaConnectionError, OllamaModelError):
            return None

    def close(self) -> None:
        """Close client connections."""
        # Ollama clients don't require explicit closing
        # but we can reset the instances
        self._sync_client = None
        self._async_client = None
        logger.info("Ollama client connections closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.close()
