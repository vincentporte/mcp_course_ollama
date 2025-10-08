"""Ollama configuration management for MCP Course.

This module provides configuration classes and validation for Ollama integration,
ensuring privacy-by-design principles and proper local LLM setup.
"""

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field, HttpUrl, validator


class OllamaConfigModel(BaseModel):
    """Pydantic model for Ollama configuration validation."""

    model_name: str = Field(..., min_length=1, description="Name of the Ollama model to use")
    endpoint: HttpUrl = Field(default="http://localhost:11434", description="Ollama server endpoint")
    timeout: int = Field(default=30, ge=1, le=300, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum number of retry attempts")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Model parameters")
    privacy_settings: dict[str, Any] = Field(default_factory=dict, description="Privacy configuration")

    @validator('endpoint')
    def validate_endpoint(cls, v):
        """Validate that endpoint is localhost for privacy."""
        parsed = urlparse(str(v))
        if parsed.hostname not in ['localhost', '127.0.0.1', '::1']:
            raise ValueError("Endpoint must be localhost for privacy-by-design compliance")
        return v

    @validator('parameters', pre=True, always=True)
    def set_default_parameters(cls, v):
        """Set default model parameters if not provided."""
        defaults = {
            "temperature": 0.7,
            "max_tokens": 2048,
            "context_window": 4096,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1
        }
        if not v:
            return defaults
        # Merge with defaults, keeping user values
        return {**defaults, **v}

    @validator('privacy_settings', pre=True, always=True)
    def set_privacy_settings(cls, v):
        """Ensure privacy-by-design settings are enforced."""
        defaults = {
            "local_only": True,
            "data_retention": "none",
            "telemetry_enabled": False,
            "external_requests": False,
            "log_prompts": False
        }
        if not v:
            return defaults
        # Enforce privacy settings
        enforced = {**v, **defaults}
        return enforced


@dataclass
class OllamaConfig:
    """Configuration class for Ollama integration.

    This class provides a convenient interface for managing Ollama configuration
    while ensuring privacy-by-design principles are maintained.

    Attributes:
        model_name: Name of the Ollama model to use
        endpoint: Ollama server endpoint (must be localhost)
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        parameters: Model generation parameters
        privacy_settings: Privacy and security configuration
    """

    model_name: str
    endpoint: str = "http://localhost:11434"
    timeout: int = 30
    max_retries: int = 3
    parameters: dict[str, Any] = field(default_factory=dict)
    privacy_settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Use Pydantic model for validation
        validated = OllamaConfigModel(
            model_name=self.model_name,
            endpoint=self.endpoint,
            timeout=self.timeout,
            max_retries=self.max_retries,
            parameters=self.parameters,
            privacy_settings=self.privacy_settings
        )

        # Update attributes with validated values
        self.model_name = validated.model_name
        self.endpoint = str(validated.endpoint)
        self.timeout = validated.timeout
        self.max_retries = validated.max_retries
        self.parameters = validated.parameters
        self.privacy_settings = validated.privacy_settings

    @classmethod
    def create_default(cls, model_name: str) -> "OllamaConfig":
        """Create a default configuration for a given model.

        Args:
            model_name: Name of the Ollama model to use

        Returns:
            OllamaConfig instance with default settings
        """
        return cls(model_name=model_name)

    @classmethod
    def create_for_course(cls, model_name: str = "llama3.2:3b") -> "OllamaConfig":
        """Create a configuration optimized for course usage.

        Args:
            model_name: Name of the Ollama model (default: llama3.2:3b)

        Returns:
            OllamaConfig instance optimized for educational use
        """
        course_parameters = {
            "temperature": 0.3,  # Lower temperature for more consistent responses
            "max_tokens": 1024,  # Reasonable limit for course interactions
            "context_window": 2048,
            "top_p": 0.8,
            "repeat_penalty": 1.05
        }

        return cls(
            model_name=model_name,
            parameters=course_parameters,
            timeout=45  # Slightly longer timeout for educational content
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        return {
            "model_name": self.model_name,
            "endpoint": self.endpoint,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "parameters": self.parameters.copy(),
            "privacy_settings": self.privacy_settings.copy()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OllamaConfig":
        """Create configuration from dictionary.

        Args:
            data: Dictionary containing configuration data

        Returns:
            OllamaConfig instance
        """
        return cls(**data)

    def is_privacy_compliant(self) -> bool:
        """Check if configuration meets privacy requirements.

        Returns:
            True if configuration is privacy compliant
        """
        # Check endpoint is localhost
        parsed = urlparse(self.endpoint)
        if parsed.hostname not in ['localhost', '127.0.0.1', '::1']:
            return False

        # Check privacy settings
        required_settings = {
            "local_only": True,
            "external_requests": False
        }

        for key, expected_value in required_settings.items():
            if self.privacy_settings.get(key) != expected_value:
                return False

        return True

    def validate(self) -> None:
        """Validate the current configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.is_privacy_compliant():
            raise ValueError("Configuration does not meet privacy requirements")

        if not self.model_name:
            raise ValueError("Model name cannot be empty")

        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")

        if self.max_retries < 0:
            raise ValueError("Max retries cannot be negative")
