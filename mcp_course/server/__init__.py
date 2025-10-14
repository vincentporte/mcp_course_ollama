"""MCP Server implementation tutorials and examples."""

from mcp_course.server.basic import BasicMCPServer
from mcp_course.server.resources import (
    APIResourceProvider,
    FileResourceProvider,
    MemoryResourceProvider,
    ResourceMetadata,
    ResourceProvider,
    ResourceRegistry,
    create_example_resources,
)
from mcp_course.server.scaffolding import ServerConfig, create_server_scaffold
from mcp_course.server.tools import ToolBuilder, ToolRegistry, create_parameter, tool


__all__ = [
    "APIResourceProvider",
    "BasicMCPServer",
    "FileResourceProvider",
    "MemoryResourceProvider",
    "ResourceMetadata",
    "ResourceProvider",
    "ResourceRegistry",
    "ServerConfig",
    "ToolBuilder",
    "ToolRegistry",
    "create_example_resources",
    "create_parameter",
    "create_server_scaffold",
    "tool"
]
