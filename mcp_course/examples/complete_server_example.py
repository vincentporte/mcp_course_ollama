#!/usr/bin/env python3
"""
Complete MCP Server Example

This example demonstrates a full-featured MCP Server that combines:
- Tools creation and registration system
- Resources exposure and management
- Proper error handling and logging
- Integration with the MCP protocol

Run this example:
    python -m mcp_course.examples.complete_server_example
"""
import asyncio
import json
import logging
from pathlib import Path
import sys

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import EmbeddedResource, ImageContent, Resource, TextContent, Tool

from mcp_course.server.resources import (
    FileResourceProvider,
    MemoryResourceProvider,
    ResourceMetadata,
    ResourceRegistry,
)
from mcp_course.server.scaffolding import ServerConfig
from mcp_course.server.tools import ToolBuilder, ToolRegistry, create_parameter, tool


class CompleteMCPServer:
    """
    Complete MCP Server implementation with tools and resources.

    This server demonstrates:
    - Full integration of tools and resources
    - Educational content for learning MCP
    - Best practices for server implementation
    - Comprehensive error handling and logging
    """

    def __init__(self, config: ServerConfig):
        """Initialize the complete MCP Server."""
        self.config = config
        self.server = Server(config.name)
        self.tool_registry = ToolRegistry()
        self.resource_registry = ResourceRegistry()
        self.logger = logging.getLogger(config.name)

        self._setup_logging()
        self._setup_tools()
        self._setup_resources()
        self._register_handlers()

    def _setup_logging(self):
        """Configure logging for the server."""
        logging.basicConfig(
            level=getattr(logging, self.config.logging_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger.info(f"Initialized {self.config.name} v{self.config.version}")

    def _setup_tools(self):
        """Set up educational and utility tools."""

        # Tool 1: MCP Protocol Explorer
        protocol_tool = (ToolBuilder("explore_mcp_protocol", "Explore MCP protocol concepts and examples")
                        .add_string_param("concept", "MCP concept to explore",
                                        enum_values=["server", "client", "tools", "resources", "prompts", "transport"])
                        .add_boolean_param("include_examples", "Include code examples", required=False, default=True)
                        .build())

        async def explore_mcp_protocol(concept: str, include_examples: bool = True) -> str:
            """Provide educational content about MCP concepts."""
            concepts = {
                "server": {
                    "description": "MCP Server exposes tools, resources, and prompts to clients",
                    "key_points": [
                        "Implements JSON-RPC protocol handlers",
                        "Manages tool execution and resource access",
                        "Provides security and access control",
                        "Supports multiple transport layers"
                    ],
                    "example": '''
# Basic MCP Server
from mcp import server

server = Server("my-server")

@server.list_tools()
async def handle_list_tools():
    return [Tool(name="example", description="Example tool")]
                    '''
                },
                "client": {
                    "description": "MCP Client connects to servers and facilitates LLM interactions",
                    "key_points": [
                        "Discovers and connects to MCP Servers",
                        "Translates LLM requests to MCP protocol",
                        "Handles server responses and errors",
                        "Manages multiple server connections"
                    ],
                    "example": '''
# Basic MCP Client usage
from mcp import ClientSession

async with ClientSession() as session:
    tools = await session.list_tools()
    result = await session.call_tool("tool_name", {"arg": "value"})
                    '''
                },
                "tools": {
                    "description": "Tools are functions that LLMs can call to perform actions",
                    "key_points": [
                        "Defined with JSON Schema for parameters",
                        "Execute arbitrary code with validation",
                        "Return structured responses",
                        "Support complex parameter types"
                    ],
                    "example": '''
# Tool definition and handler
@server.call_tool()
async def handle_call_tool(name: str, arguments: dict):
    if name == "calculator":
        return [TextContent(type="text", text=str(eval(arguments["expression"])))]
                    '''
                },
                "resources": {
                    "description": "Resources provide structured data access to LLMs",
                    "key_points": [
                        "Identified by URI schemes",
                        "Support multiple content types",
                        "Enable data discovery and access",
                        "Provide metadata and caching"
                    ],
                    "example": '''
# Resource handler
@server.read_resource()
async def handle_read_resource(uri: str):
    if uri.startswith("file://"):
        content = Path(uri[7:]).read_text()
        return TextContent(type="text", text=content)
                    '''
                },
                "prompts": {
                    "description": "Prompts are reusable templates for common LLM interactions",
                    "key_points": [
                        "Parameterized prompt templates",
                        "Consistent interaction patterns",
                        "Reduce prompt engineering overhead",
                        "Enable prompt sharing and reuse"
                    ],
                    "example": '''
# Prompt definition
@server.list_prompts()
async def handle_list_prompts():
    return [Prompt(
        name="code_review",
        description="Review code for best practices",
        arguments=[PromptArgument(name="code", description="Code to review")]
    )]
                    '''
                },
                "transport": {
                    "description": "Transport layers handle communication between clients and servers",
                    "key_points": [
                        "Stdio: Standard input/output streams",
                        "HTTP: RESTful API endpoints",
                        "WebSocket: Real-time bidirectional communication",
                        "Custom: Application-specific protocols"
                    ],
                    "example": '''
# Stdio transport (most common)
async with stdio_server() as (read_stream, write_stream):
    await server.run(read_stream, write_stream, init_options)
                    '''
                }
            }

            if concept not in concepts:
                return f"Unknown concept: {concept}. Available: {list(concepts.keys())}"

            info = concepts[concept]
            result = f"# MCP {concept.title()}\n\n{info['description']}\n\n## Key Points:\n"

            for point in info['key_points']:
                result += f"- {point}\n"

            if include_examples and 'example' in info:
                result += f"\n## Example:\n```python{info['example']}\n```"

            return result

        self.tool_registry.register_tool(protocol_tool, explore_mcp_protocol)

        # Tool 2: Resource Manager
        @tool("manage_resources", "Manage and explore available resources", [
            create_parameter("action", "string", "Action to perform",
                           enum_values=["list", "info", "search", "content"]),
            create_parameter("uri", "string", "Resource URI (for info/content actions)", required=False),
            create_parameter("query", "string", "Search query (for search action)", required=False)
        ], self.tool_registry)
        async def manage_resources(action: str, uri: str | None = None, query: str | None = None) -> str:
            """Manage and explore resources."""
            result = ""

            if action == "list":
                result = await self._handle_list_resources()
            elif action == "info":
                result = await self._handle_resource_info(uri)
            elif action == "content":
                result = await self._handle_resource_content(uri)
            elif action == "search":
                result = await self._handle_resource_search(query)
            else:
                result = f"Unknown action: {action}"

            return result

        async def _handle_list_resources(self) -> str:
            """Handle listing all resources."""
            resources = await self.resource_registry.list_resources()
            result = "Available Resources:\n"
            for resource in resources:
                result += f"- {resource.name} ({resource.uri})\n"
                result += f"  Type: {resource.mimeType}\n"
                result += f"  Description: {resource.description}\n\n"
            return result

        async def _handle_resource_info(self, uri: str | None = None) -> str:
            """Handle getting resource information."""
            if not uri:
                return "URI required for info action"
            try:
                metadata = await self.resource_registry.get_resource_metadata(uri)
                return f"""Resource Information:
                        URI: {metadata.uri}
                        Name: {metadata.name}
                        Description: {metadata.description}
                        MIME Type: {metadata.mime_type}
                        Size: {metadata.size} bytes
                        Tags: {metadata.tags}
                        """
            except Exception as e:
                return f"Error getting resource info: {e}"

        async def _handle_resource_content(self, uri: str | None = None) -> str:
            """Handle getting resource content."""
            if not uri:
                return "URI required for content action"
            try:
                content = await self.resource_registry.get_resource_content(uri)
                preview = content.text[:500] + "..." if len(content.text) > 500 else content.text
                return f"Resource Content ({uri}):\n\n{preview}"
            except Exception as e:
                return f"Error getting resource content: {e}"

        async def _handle_resource_search(self, query: str | None = None) -> str:
            """Handle searching resources."""
            if not query:
                return "Query required for search action"
            resources = await self.resource_registry.list_resources(query)
            result = f"Search Results for '{query}':\n"
            for resource in resources:
                if query.lower() in resource.name.lower() or query.lower() in resource.description.lower():
                    result += f"- {resource.name} ({resource.uri})\n"
            return result

        # Tool 3: Server Information
        server_info_tool = (ToolBuilder("server_info", "Get information about this MCP Server")
                           .add_string_param("info_type", "Type of information",
                                           enum_values=["status", "capabilities", "tools", "resources", "stats"])
                           .build())

        async def server_info_handler(info_type: str) -> str:
            """Provide server information."""
            if info_type == "status":
                return f"""Server Status:
                        Name: {self.config.name}
                        Version: {self.config.version}
                        Description: {self.config.description}
                        Status: Running
                        """

            elif info_type == "capabilities":
                return f"Server Capabilities:\n{json.dumps(self.config.capabilities, indent=2)}"

            elif info_type == "tools":
                tools = self.tool_registry.get_tool_names()
                return f"Available Tools ({len(tools)}):\n" + "\n".join(f"- {tool}" for tool in tools)

            elif info_type == "resources":
                resources = await self.resource_registry.list_resources()
                return f"Available Resources ({len(resources)}):\n" + "\n".join(f"- {r.name}" for r in resources)

            elif info_type == "stats":
                tool_count = len(self.tool_registry.get_tool_names())
                resource_count = len(await self.resource_registry.list_resources())
                return f"""Server Statistics:
                        Tools: {tool_count}
                        Resources: {resource_count}
                        Providers: {len(self.resource_registry.providers)}
                        """

            else:
                return f"Unknown info type: {info_type}"

        self.tool_registry.register_tool(server_info_tool, server_info_handler)

        self.logger.info(f"Registered {len(self.tool_registry.get_tool_names())} tools")

    def _setup_resources(self):
        """Set up educational resources."""

        # Add file provider
        file_provider = FileResourceProvider(
            base_path=Path.cwd(),
            allowed_extensions=['.txt', '.md', '.json', '.py', '.yaml']
        )
        self.resource_registry.add_provider(file_provider)

        # Add memory provider with educational content
        memory_provider = MemoryResourceProvider()

        # MCP Tutorial content
        memory_provider.add_resource(
            uri="memory://tutorial/getting-started",
            content="""# Getting Started with MCP

                    ## What is MCP?

                    The Model Context Protocol (MCP) is an open standard that enables secure, controlled interactions between AI models and external systems. It provides a standardized way for Large Language Models (LLMs) to access tools, resources, and prompts.

                    ## Key Benefits

                    1. **Standardization**: Consistent interface across different implementations
                    2. **Security**: Controlled access to external systems
                    3. **Extensibility**: Easy to add new capabilities
                    4. **Privacy**: Can be deployed locally without external dependencies

                    ## Core Concepts

                    ### Servers
                    MCP Servers expose functionality to clients. They can provide:
                    - **Tools**: Functions that can be called
                    - **Resources**: Data that can be read
                    - **Prompts**: Template prompts for common tasks

                    ### Clients
                    MCP Clients connect to servers and facilitate interactions with LLMs.

                    ### Transport
                    Communication happens over various transports:
                    - Stdio (most common)
                    - HTTP
                    - WebSocket

                    ## Getting Started

                    1. Install the MCP Python package: `pip install mcp`
                    2. Create a basic server
                    3. Register tools and resources
                    4. Run the server
                    5. Connect a client

                    This tutorial server demonstrates all these concepts!
                    """,
            name="Getting Started Guide",
            description="Introduction to MCP concepts and implementation",
            mime_type="text/markdown"
        )

        # Implementation examples
        memory_provider.add_resource(
            uri="memory://examples/basic-server",
            content=json.dumps({
                "title": "Basic MCP Server Implementation",
                "description": "Complete example of a basic MCP Server",
                "code": """#!/usr/bin/env python3
                        import asyncio
                        from mcp import server
                        from mcp.server.stdio import stdio_server
                        from mcp.server.models import InitializationOptions
                        from mcp.types import Tool, TextContent

                        # Create server
                        server = Server("basic-server")

                        @server.list_tools()
                        async def handle_list_tools():
                            return [
                                Tool(
                                    name="echo",
                                    description="Echo back a message",
                                    inputSchema={
                                        "type": "object",
                                        "properties": {
                                            "message": {"type": "string"}
                                        },
                                        "required": ["message"]
                                    }
                                )
                            ]

                        @server.call_tool()
                        async def handle_call_tool(name: str, arguments: dict):
                            if name == "echo":
                                return [TextContent(type="text", text=f"Echo: {arguments['message']}")]

                        async def main():
                            async with stdio_server() as (read_stream, write_stream):
                                await server.run(
                                    read_stream,
                                    write_stream,
                                    InitializationOptions(
                                        server_name="basic-server",
                                        server_version="1.0.0",
                                        capabilities={"tools": {}}
                                    )
                                )

                        if __name__ == "__main__":
                            asyncio.run(main())
                        """,
                "explanation": "This example shows the minimal code needed to create a functional MCP Server with one tool."
            }, indent=2),
            name="Basic Server Example",
            description="Complete code example for a basic MCP Server",
            mime_type="application/json"
        )

        self.resource_registry.add_provider(memory_provider)

        # Register static resources
        static_resources = [
            ResourceMetadata(
                uri="memory://tutorial/getting-started",
                name="Getting Started Guide",
                description="Introduction to MCP concepts and implementation",
                mime_type="text/markdown",
                tags=["tutorial", "introduction", "guide"]
            ),
            ResourceMetadata(
                uri="memory://examples/basic-server",
                name="Basic Server Example",
                description="Complete code example for a basic MCP Server",
                mime_type="application/json",
                tags=["example", "code", "server"]
            )
        ]

        for resource_meta in static_resources:
            self.resource_registry.add_static_resource(resource_meta)

        self.logger.info("Educational resources configured")

    def _register_handlers(self):
        """Register MCP protocol handlers."""

        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """Handle list_tools requests."""
            self.logger.info("Received list_tools request")
            tools = self.tool_registry.get_all_tools()
            self.logger.info(f"Returning {len(tools)} tools")
            return tools

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
            """Handle call_tool requests."""
            self.logger.info(f"Received call_tool request: {name}")

            try:
                result = await self.tool_registry.execute_tool(name, arguments)
                self.logger.info(f"Tool {name} executed successfully")
                return result
            except Exception as e:
                error_msg = f"Error executing tool {name}: {e!s}"
                self.logger.error(error_msg)
                return [TextContent(type="text", text=error_msg)]

        @self.server.list_resources()
        async def handle_list_resources() -> list[Resource]:
            """Handle list_resources requests."""
            self.logger.info("Received list_resources request")

            try:
                resources = await self.resource_registry.list_resources()
                self.logger.info(f"Returning {len(resources)} resources")
                return resources
            except Exception as e:
                self.logger.error(f"Error listing resources: {e}")
                return []

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> TextContent | ImageContent | EmbeddedResource:
            """Handle read_resource requests."""
            self.logger.info(f"Received read_resource request for: {uri}")

            try:
                content = await self.resource_registry.get_resource_content(uri)
                self.logger.info(f"Successfully retrieved content for: {uri}")
                return content
            except Exception as e:
                error_msg = f"Error reading resource {uri}: {e!s}"
                self.logger.error(error_msg)
                return TextContent(type="text", text=error_msg)

    async def run(self):
        """Run the complete MCP Server."""
        try:
            self.logger.info("Starting Complete MCP Server with stdio transport")

            async with stdio_server() as (read_stream, write_stream):
                init_options = InitializationOptions(
                    server_name=self.config.name,
                    server_version=self.config.version,
                    capabilities=self.config.capabilities
                )

                await self.server.run(read_stream, write_stream, init_options)

        except Exception as e:
            self.logger.error(f"Server error: {e}")
            raise
        finally:
            self.logger.info("Complete MCP Server shutdown")


async def demonstrate_complete_server():
    """Demonstrate the complete server capabilities."""
    print("=== Complete MCP Server Demonstration ===")
    print()

    config = ServerConfig(
        name="complete-tutorial-server",
        version="1.0.0",
        description="Complete MCP Server with tools and resources for learning",
        capabilities={
            "tools": {},
            "resources": {},
            "prompts": {}
        }
    )

    server = CompleteMCPServer(config)

    print(f"Server: {config.name} v{config.version}")
    print(f"Description: {config.description}")
    print()

    # Show capabilities
    print("Available Tools:")
    for tool_name in server.tool_registry.get_tool_names():
        print(f"- {tool_name}")
    print()

    print("Available Resources:")
    resources = await server.resource_registry.list_resources()
    for resource in resources:
        print(f"- {resource.name} ({resource.uri})")
    print()

    # Test some functionality
    print("Testing Tool Execution:")
    try:
        result = await server.tool_registry.execute_tool("explore_mcp_protocol", {
            "concept": "server",
            "include_examples": False
        })
        print(f"✅ MCP Protocol Explorer: {result[0].text[:100]}...")
    except Exception as e:
        print(f"❌ Error: {e}")

    print()
    print("This complete server demonstrates:")
    print("- Tool creation and registration")
    print("- Resource management and exposure")
    print("- Educational content delivery")
    print("- Proper error handling and logging")
    print("- Full MCP protocol implementation")


async def create_and_run_complete_server():
    """Create and run the complete server."""
    config = ServerConfig(
        name="complete-tutorial-server",
        version="1.0.0",
        description="Complete MCP Server with tools and resources for learning"
    )

    server = CompleteMCPServer(config)
    await server.run()


async def main():
    """Main entry point."""

    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        await create_and_run_complete_server()
    else:
        await demonstrate_complete_server()


if __name__ == "__main__":
    asyncio.run(main())
