"""Validation utilities for course components."""

from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

from mcp_course.config.settings import (
    DURATION_MAX,
    DURATION_MIN,
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
)


class ValidationResult:
    """Result of a validation operation."""

    def __init__(self, is_valid: bool, errors: list[str] | None = None):
        self.is_valid = is_valid
        self.errors = errors or []

    def __bool__(self) -> bool:
        return self.is_valid

    def __str__(self) -> str:
        if self.is_valid:
            return "Validation passed"
        return f"Validation failed: {'; '.join(self.errors)}"


def validate_url(url: str) -> ValidationResult:
    """Validate a URL format.

    Args:
        url: URL to validate

    Returns:
        ValidationResult indicating if URL is valid
    """
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return ValidationResult(False, ["Invalid URL format"])
        return ValidationResult(True)
    except Exception as e:
        return ValidationResult(False, [f"URL validation error: {e!s}"])


def validate_model_name(model_name: str) -> ValidationResult:
    """Validate Ollama model name format.

    Args:
        model_name: Model name to validate

    Returns:
        ValidationResult indicating if model name is valid
    """
    # Basic pattern: name:tag or just name
    pattern = r'^[a-zA-Z0-9._-]+(?::[a-zA-Z0-9._-]+)?$'

    if not model_name:
        return ValidationResult(False, ["Model name cannot be empty"])

    if not re.match(pattern, model_name):
        return ValidationResult(False, ["Invalid model name format"])

    return ValidationResult(True)


def validate_temperature(temperature: float) -> ValidationResult:
    """Validate temperature parameter.

    Args:
        temperature: Temperature value to validate

    Returns:
        ValidationResult indicating if temperature is valid
    """
    if not isinstance(temperature, (int, float)):
        return ValidationResult(False, ["Temperature must be a number"])

    if temperature < TEMPERATURE_MIN or temperature > TEMPERATURE_MAX:
        return ValidationResult(False, [f"Temperature must be between {TEMPERATURE_MIN} and {TEMPERATURE_MAX}"])

    return ValidationResult(True)


def validate_file_path(file_path: str | Path, must_exist: bool = False) -> ValidationResult:
    """Validate file path.

    Args:
        file_path: Path to validate
        must_exist: Whether the file must already exist

    Returns:
        ValidationResult indicating if path is valid
    """
    try:
        path = Path(file_path)

        if must_exist and not path.exists():
            return ValidationResult(False, [f"File does not exist: {path}"])

        # Check if parent directory is writable (for new files)
        if not must_exist and not path.exists():
            parent = path.parent
            if not parent.exists():
                try:
                    parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    return ValidationResult(False, [f"Cannot create parent directory: {e}"])

        return ValidationResult(True)

    except Exception as e:
        return ValidationResult(False, [f"Path validation error: {e!s}"])


def validate_config_dict(config: dict[str, Any], required_keys: list[str]) -> ValidationResult:
    """Validate configuration dictionary.

    Args:
        config: Configuration dictionary to validate
        required_keys: List of required keys

    Returns:
        ValidationResult indicating if configuration is valid
    """
    errors = []

    if not isinstance(config, dict):
        return ValidationResult(False, ["Configuration must be a dictionary"])

    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        errors.append(f"Missing required keys: {', '.join(missing_keys)}")

    return ValidationResult(len(errors) == 0, errors)


class CourseValidator:
    """Validator for course-specific data structures."""

    @staticmethod
    def validate_module_id(module_id: str) -> ValidationResult:
        """Validate module ID format."""
        pattern = r'^[a-z][a-z0-9_]*$'

        if not module_id:
            return ValidationResult(False, ["Module ID cannot be empty"])

        if not re.match(pattern, module_id):
            return ValidationResult(False, ["Module ID must start with letter and contain only lowercase letters, numbers, and underscores"])

        return ValidationResult(True)

    @staticmethod
    def validate_difficulty_level(difficulty: str) -> ValidationResult:
        """Validate difficulty level."""
        valid_levels = ["beginner", "intermediate", "advanced"]

        if difficulty not in valid_levels:
            return ValidationResult(False, [f"Difficulty must be one of: {', '.join(valid_levels)}"])

        return ValidationResult(True)

    @staticmethod
    def validate_duration(duration: int) -> ValidationResult:
        """Validate estimated duration in minutes."""
        if not isinstance(duration, int):
            return ValidationResult(False, ["Duration must be an integer"])

        if duration <= DURATION_MIN:
            return ValidationResult(False, ["Duration must be positive"])

        if duration > DURATION_MAX:  # 8 hours
            return ValidationResult(False, [f"Duration should not exceed {DURATION_MAX} hours ({DURATION_MAX * 60} minutes)"])

        return ValidationResult(True)
