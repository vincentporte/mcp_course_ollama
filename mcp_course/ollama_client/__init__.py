"""Ollama client integration module for MCP Course.

This module provides configuration, connection management, and utilities
for integrating Ollama with the MCP course platform.
"""

from mcp_course.ollama_client.client import OllamaClient
from mcp_course.ollama_client.config import OllamaConfig
from mcp_course.ollama_client.health import OllamaHealthChecker
from mcp_course.ollama_client.models import OllamaModelManager
from mcp_course.ollama_client.performance import OllamaPerformanceTester
from mcp_course.ollama_client.setup import OllamaSetupVerifier


__all__ = [
    "OllamaClient",
    "OllamaConfig",
    "OllamaHealthChecker",
    "OllamaModelManager",
    "OllamaPerformanceTester",
    "OllamaSetupVerifier"
]
