"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path
import tempfile
import shutil

from mcp_course.config.settings import ConfigManager, OllamaConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def config_manager(temp_dir):
    """Create a test configuration manager."""
    config_path = temp_dir / "test_config.json"
    return ConfigManager(config_path)


@pytest.fixture
def ollama_config():
    """Create a test Ollama configuration."""
    return OllamaConfig(
        endpoint="http://localhost:11434",
        default_model="llama3.2:3b",
        temperature=0.7
    )