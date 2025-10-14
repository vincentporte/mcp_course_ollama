#!/usr/bin/env python3
"""
Basic MCP Server Example

This example demonstrates how to create a minimal MCP Server using
the scaffolding utilities provided in the course framework.

Run this example:
    python -m mcp_course.examples.basic_server_example
"""

import asyncio

from mcp_course.server.basic import BasicMCPServer
from mcp_course.server.scaffolding import ServerConfig


async def main():
    """Run a basic MCP Server example."""
    print("=== Basic MCP Server Example ===")
    print("This example shows how to create and configure a basic MCP Server")
    print()

    # Create server configuration
    config = ServerConfig(
        name="example-server",
        version="1.0.0",
        description="Example MCP Server for demonstration purposes",
        logging_level="INFO"
    )

    print(f"Server Name: {config.name}")
    print(f"Version: {config.version}")
    print(f"Description: {config.description}")
    print()

    # Create server instance
    server = BasicMCPServer(config) # noqa F821

    print("Server created successfully!")
    print("Available tools will include: echo, get_time, calculate")
    print()
    print("To run this server, it would normally be started with:")
    print("python -m mcp_course.examples.basic_server_example --run")
    print()
    print("The server would then wait for MCP Client connections via stdio.")

    # In a real scenario, you would call:
    # await server.run() # noqa ERA001
    # But for this example, we'll just show the setup


if __name__ == "__main__":
    asyncio.run(main())
