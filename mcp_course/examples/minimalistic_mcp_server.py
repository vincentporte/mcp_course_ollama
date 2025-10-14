#!/usr/bin/env python3
"""
Basic MCP Server Template

This template demonstrates the minimal structure needed to create
a functional MCP Server using the Python mcp package.
"""

import asyncio
import logging

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server


async def main():
    """Main entry point for the MCP Server."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("basic-mcp-server")

    # Create server instance
    server = Server("basic-mcp-server")

    # TODO: Add your tools and resources here

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
