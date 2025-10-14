"""Basic MCP Server implementation with request/response handling examples."""

import asyncio
from datetime import datetime
import logging
from typing import Any

from mcp import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
    Tool,
)

from mcp_course.server.scaffolding import ServerConfig


class BasicMCPServer:
    """
    A basic MCP Server implementation demonstrating core patterns.

    This class shows how to:
    - Initialize an MCP Server with proper configuration
    - Handle basic request/response patterns
    - Implement server lifecycle management
    - Provide logging and error handling
    """

    def __init__(self, config: ServerConfig):
        """Initialize the basic MCP Server."""
        self.config = config
        self.server = Server(config.name)
        self.logger = logging.getLogger(config.name)
        self._setup_logging()
        self._register_handlers()

    def _setup_logging(self) -> None:
        """Configure logging for the server."""
        logging.basicConfig(
            level=getattr(logging, self.config.logging_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger.info(f"Initialized {self.config.name} v{self.config.version}")

    def _register_handlers(self) -> None:
        """Register basic request handlers."""

        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """
            Handle list_tools requests.

            This is a fundamental MCP Server capability that tells clients
            what tools are available for use.
            """
            self.logger.info("Received list_tools request")

            tools = [
                Tool(
                    name="echo",
                    description="Echo back the provided message",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Message to echo back"
                            }
                        },
                        "required": ["message"]
                    }
                ),
                Tool(
                    name="get_time",
                    description="Get the current server time",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False
                    }
                ),
                Tool(
                    name="calculate",
                    description="Perform basic arithmetic calculations",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["add", "subtract", "multiply", "divide"],
                                "description": "The arithmetic operation to perform"
                            },
                            "a": {
                                "type": "number",
                                "description": "First number"
                            },
                            "b": {
                                "type": "number",
                                "description": "Second number"
                            }
                        },
                        "required": ["operation", "a", "b"]
                    }
                )
            ]

            self.logger.info(f"Returning {len(tools)} available tools")
            return tools

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """
            Handle call_tool requests.

            This demonstrates the basic pattern for implementing tool execution
            with proper error handling and response formatting.
            """
            self.logger.info(f"Received call_tool request: {name} with args {arguments}")

            try:
                if name == "echo":
                    message = arguments.get("message", "")
                    result = f"Echo: {message}"

                elif name == "get_time":
                    current_time = datetime.now().isoformat()
                    result = f"Current server time: {current_time}"

                elif name == "calculate":
                    operation = arguments["operation"]
                    a = float(arguments["a"])
                    b = float(arguments["b"])

                    if operation == "add":
                        calc_result = a + b
                    elif operation == "subtract":
                        calc_result = a - b
                    elif operation == "multiply":
                        calc_result = a * b
                    elif operation == "divide":
                        if b == 0:
                            raise ValueError("Division by zero is not allowed")
                        calc_result = a / b
                    else:
                        raise ValueError(f"Unknown operation: {operation}")

                    result = f"{a} {operation} {b} = {calc_result}"

                else:
                    raise ValueError(f"Unknown tool: {name}")

                self.logger.info(f"Tool {name} executed successfully")
                return [TextContent(type="text", text=result)]

            except Exception as e:
                error_msg = f"Error executing tool {name}: {e!s}"
                self.logger.error(error_msg)
                return [TextContent(type="text", text=error_msg)]

    async def run(self) -> None:
        """
        Run the MCP Server with stdio transport.

        This method demonstrates the standard pattern for running an MCP Server
        that communicates via standard input/output streams.
        """
        try:
            self.logger.info("Starting MCP Server with stdio transport")

            async with stdio_server() as (read_stream, write_stream):
                # Set up initialization options
                init_options = InitializationOptions(
                    server_name=self.config.name,
                    server_version=self.config.version,
                    capabilities=self.config.capabilities
                )

                # Run the server
                await self.server.run(
                    read_stream,
                    write_stream,
                    init_options
                )

        except Exception as e:
            self.logger.error(f"Server error: {e}")
            raise
        finally:
            self.logger.info("MCP Server shutdown complete")


async def create_and_run_basic_server() -> None:
    """
    Convenience function to create and run a basic MCP Server.

    This demonstrates the complete lifecycle of an MCP Server from
    configuration to execution.
    """
    # Create server configuration
    config = ServerConfig(
        name="basic-tutorial-server",
        version="1.0.0",
        description="Basic MCP Server for learning fundamental patterns",
        capabilities={
            "tools": {},
            "resources": {},
            "prompts": {}
        }
    )

    # Create and run server
    server = BasicMCPServer(config)
    await server.run()


def demonstrate_request_response_patterns() -> dict[str, Any]:
    """
    Demonstrate common MCP request/response patterns.

    This function shows examples of the JSON-RPC messages that flow
    between MCP Clients and Servers.

    Returns:
        Dictionary containing example request/response pairs
    """
    examples = {
        "list_tools_request": {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        },
        "list_tools_response": {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echo back the provided message",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "message": {
                                    "type": "string",
                                    "description": "Message to echo back"
                                }
                            },
                            "required": ["message"]
                        }
                    }
                ]
            }
        },
        "call_tool_request": {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "echo",
                "arguments": {
                    "message": "Hello, MCP!"
                }
            }
        },
        "call_tool_response": {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": "Echo: Hello, MCP!"
                    }
                ]
            }
        }
    }

    return examples


if __name__ == "__main__":
    # Run the basic server when executed directly
    asyncio.run(create_and_run_basic_server())
