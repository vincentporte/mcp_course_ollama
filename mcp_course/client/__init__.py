"""MCP Client implementation framework for educational purposes.

This module provides a comprehensive MCP Client implementation that demonstrates
how to connect to MCP Servers, discover capabilities, and interact with tools
and resources. It includes error handling, connection management, and integration
patterns with Ollama for LLM interactions.
"""

from mcp_course.client.basic import BasicMCPClient
from mcp_course.client.connection import ConnectionManager, ServerDiscovery
from mcp_course.client.conversation import ConversationManager
from mcp_course.client.integration import OllamaMCPBridge
from mcp_course.client.prompts import PromptEngineering


__all__ = [
    "BasicMCPClient",
    "ConnectionManager",
    "ConversationManager",
    "OllamaMCPBridge",
    "PromptEngineering",
    "ServerDiscovery"
]
