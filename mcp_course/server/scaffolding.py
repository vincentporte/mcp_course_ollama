"""MCP Server scaffolding and configuration patterns."""

from dataclasses import dataclass, field
import logging
from typing import Any

from mcp import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server


@dataclass
class ServerConfig:
    """Configuration for MCP Server initialization."""

    name: str
    version: str = "1.0.0"
    description: str = ""
    capabilities: dict[str, Any] = field(default_factory=dict)
    logging_level: str = "INFO"

    def __post_init__(self):
        """Set default capabilities if not provided."""
        if not self.capabilities:
            self.capabilities = {
                "tools": {},
                "resources": {},
                "prompts": {}
            }


def create_server_scaffold(config: ServerConfig) -> Server:
    """
    Create a basic MCP Server scaffold with standard configuration.

    This function demonstrates the fundamental pattern for initializing
    an MCP Server with proper configuration and logging setup.

    Args:
        config: Server configuration containing name, version, and capabilities

    Returns:
        Configured MCP Server instance ready for tool and resource registration

    Example:
        >>> config = ServerConfig(
        ...     name="tutorial-server",
        ...     description="A basic MCP server for learning"
        ... )
        >>> server = create_server_scaffold(config)
    """
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.logging_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(config.name)

    # Create server instance
    server = Server(config.name)

    logger.info(f"Initializing MCP Server: {config.name} v{config.version}")
    logger.info(f"Description: {config.description}")
    logger.info(f"Capabilities: {list(config.capabilities.keys())}")

    return server


async def run_server_stdio(server: Server, config: ServerConfig) -> None:
    """
    Run the MCP Server using stdio transport.

    This is the standard way to run an MCP Server that communicates
    via standard input/output streams.

    Args:
        server: Configured MCP Server instance
        config: Server configuration for initialization options

    Example:
        >>> server = create_server_scaffold(config)
        >>> await run_server_stdio(server, config)
    """
    logger = logging.getLogger(config.name)

    try:
        # Initialize server with stdio transport
        async with stdio_server() as (read_stream, write_stream):
            logger.info("Starting MCP Server with stdio transport")

            # Set up initialization options
            init_options = InitializationOptions(
                server_name=config.name,
                server_version=config.version,
                capabilities=config.capabilities
            )

            # Run the server
            await server.run(
                read_stream,
                write_stream,
                init_options
            )

    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        logger.info("MCP Server shutdown complete")


def create_basic_server_template() -> str:
    """
    Generate a basic MCP Server template code.

    This function returns a complete, runnable MCP Server template
    that students can use as a starting point for their own implementations.

    Returns:
        Python code string for a basic MCP Server
    """
    template = '''#!/usr/bin/env python3
"""
Basic MCP Server Template

This template demonstrates the minimal structure needed to create
a functional MCP Server using the Python mcp package.
"""

import asyncio
import logging
from mcp import server
from mcp.server.stdio import stdio_server
from mcp.server.models import InitializationOptions


async def main():
    """Main entry point for the MCP Server."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("basic-mcp-server")

    # Create server instance
    server = Server("basic-mcp-server")

    # TODO: Add your tools and resources here
    # Example:
    # @server.list_tools()
    # async def handle_list_tools():
    #     return [...]

    logger.info("Starting Basic MCP Server")

    # Run server with stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="basic-mcp-server",
                server_version="1.0.0",
                capabilities={
                    "tools": {},
                    "resources": {},
                    "prompts": {}
                }
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
'''
    return template
