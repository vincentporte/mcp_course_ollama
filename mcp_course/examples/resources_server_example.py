#!/usr/bin/env python3
"""
MCP Server with Resources Example

This example demonstrates how to create a complete MCP Server that exposes
resources using the resource management system.

Run this example:
    python -m mcp_course.examples.resources_server_example
"""

import asyncio
import json
import logging
from pathlib import Path
import sys

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import EmbeddedResource, ImageContent, Resource, TextContent

from mcp_course.server.resources import (
    APIResourceProvider,
    FileResourceProvider,
    MemoryResourceProvider,
    ResourceMetadata,
    ResourceRegistry,
    create_example_resources,
)
from mcp_course.server.scaffolding import ServerConfig


class ResourcesEnabledMCPServer:
    """
    MCP Server that demonstrates comprehensive resource integration.

    This server shows how to:
    - Use the ResourceRegistry for centralized resource management
    - Expose multiple types of resources (files, memory, API)
    - Handle resource listing and content retrieval
    - Integrate with the MCP protocol handlers
    """

    def __init__(self, config: ServerConfig):
        """Initialize the resources-enabled MCP Server."""
        self.config = config
        self.server = Server(config.name)
        self.resource_registry = ResourceRegistry()
        self.logger = logging.getLogger(config.name)

        self._setup_logging()
        self._setup_resources()
        self._register_handlers()

    def _setup_logging(self):
        """Configure logging for the server."""
        logging.basicConfig(
            level=getattr(logging, self.config.logging_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger.info(f"Initialized {self.config.name} v{self.config.version}")

    def _setup_resources(self):
        """Set up resource providers and sample resources."""

        # Add file provider for local files
        file_provider = FileResourceProvider(
            base_path=Path.cwd(),
            allowed_extensions=['.txt', '.md', '.json', '.py', '.yaml', '.yml', '.csv']
        )
        self.resource_registry.add_provider(file_provider)

        # Add memory provider with educational content
        memory_provider = MemoryResourceProvider()

        # MCP Protocol documentation
        memory_provider.add_resource(
            uri="memory://docs/mcp-protocol",
            content="""# Model Context Protocol (MCP)

MCP is a protocol that enables secure, controlled interactions between AI models and external systems.

## Key Components:

1. **MCP Server**: Exposes tools, resources, and prompts
2. **MCP Client**: Connects to servers and facilitates LLM interactions
3. **Transport Layer**: Handles communication (stdio, HTTP, WebSocket)

## Resource Types:

- **Tools**: Functions that can be called by the LLM
- **Resources**: Data sources that can be read by the LLM
- **Prompts**: Template prompts for common tasks

## Benefits:

- Standardized interface for LLM integrations
- Security through controlled access
- Extensibility via custom servers
- Privacy through local deployment options
""",
            name="MCP Protocol Overview",
            description="Comprehensive overview of the Model Context Protocol",
            mime_type="text/markdown",
            metadata={"category": "documentation", "level": "beginner"}
        )

        # Server implementation guide
        memory_provider.add_resource(
            uri="memory://guides/server-implementation",
            content=json.dumps({
                "title": "MCP Server Implementation Guide",
                "steps": [
                    {
                        "step": 1,
                        "title": "Setup Project Structure",
                        "description": "Create Python package with proper dependencies",
                        "code": "pip install mcp>=1.0.0"
                    },
                    {
                        "step": 2,
                        "title": "Create Server Instance",
                        "description": "Initialize MCP Server with configuration",
                        "code": "server = Server('my-server')"
                    },
                    {
                        "step": 3,
                        "title": "Register Handlers",
                        "description": "Add tools, resources, and prompt handlers",
                        "code": "@server.list_tools()\\nasync def handle_list_tools(): ..."
                    },
                    {
                        "step": 4,
                        "title": "Run Server",
                        "description": "Start server with stdio transport",
                        "code": "await server.run(read_stream, write_stream, init_options)"
                    }
                ],
                "best_practices": [
                    "Use proper error handling",
                    "Validate input parameters",
                    "Implement comprehensive logging",
                    "Follow security guidelines"
                ]
            }, indent=2),
            name="Server Implementation Guide",
            description="Step-by-step guide for implementing MCP Servers",
            mime_type="application/json",
            metadata={"category": "guide", "level": "intermediate"}
        )

        # Example configurations
        memory_provider.add_resource(
            uri="memory://examples/server-config",
            content="""# Example MCP Server Configuration

## Basic Configuration
```python
from mcp_course.server.scaffolding import ServerConfig

config = ServerConfig(
    name="my-mcp-server",
    version="1.0.0",
    description="My first MCP Server",
    capabilities={
        "tools": {},
        "resources": {},
        "prompts": {}
    }
)
```

## Advanced Configuration
```python
config = ServerConfig(
    name="advanced-server",
    version="2.1.0",
    description="Advanced MCP Server with full capabilities",
    capabilities={
        "tools": {
            "listChanged": True
        },
        "resources": {
            "subscribe": True,
            "listChanged": True
        },
        "prompts": {
            "listChanged": True
        }
    },
    logging_level="DEBUG"
)
```
""",
            name="Server Configuration Examples",
            description="Example configurations for MCP Servers",
            mime_type="text/markdown",
            metadata={"category": "examples", "level": "beginner"}
        )

        self.resource_registry.add_provider(memory_provider)

        # Add API provider for external resources (restricted for security)
        api_provider = APIResourceProvider(
            allowed_hosts=['httpbin.org', 'api.github.com']
        )
        self.resource_registry.add_provider(api_provider)

        # Register static resource definitions
        static_resources = [
            ResourceMetadata(
                uri="memory://docs/mcp-protocol",
                name="MCP Protocol Overview",
                description="Comprehensive overview of the Model Context Protocol",
                mime_type="text/markdown",
                tags=["documentation", "protocol", "overview"]
            ),
            ResourceMetadata(
                uri="memory://guides/server-implementation",
                name="Server Implementation Guide",
                description="Step-by-step guide for implementing MCP Servers",
                mime_type="application/json",
                tags=["guide", "implementation", "tutorial"]
            ),
            ResourceMetadata(
                uri="memory://examples/server-config",
                name="Server Configuration Examples",
                description="Example configurations for MCP Servers",
                mime_type="text/markdown",
                tags=["examples", "configuration", "setup"]
            )
        ]

        for resource_meta in static_resources:
            self.resource_registry.add_static_resource(resource_meta)

        self.logger.info("Resource providers and sample resources configured")

    def _register_handlers(self):
        """Register MCP protocol handlers for resources."""

        @self.server.list_resources()
        async def handle_list_resources() -> list[Resource]:
            """Handle list_resources requests using the resource registry."""
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
            """Handle read_resource requests using the resource registry."""
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
        """Run the MCP Server with stdio transport."""
        try:
            self.logger.info("Starting MCP Server with stdio transport")

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
            self.logger.info("MCP Server shutdown complete")


async def demonstrate_server():
    """Demonstrate the resources-enabled server without actually running it."""
    print("=== Resources-Enabled MCP Server Example ===")
    print()

    # Create server configuration
    config = ServerConfig(
        name="resources-demo-server",
        version="1.0.0",
        description="MCP Server demonstrating comprehensive resource management",
        capabilities={
            "tools": {},
            "resources": {},
            "prompts": {}
        }
    )

    # Create server
    server = ResourcesEnabledMCPServer(config)

    print(f"Server: {config.name} v{config.version}")
    print(f"Description: {config.description}")
    print()

    # Show available resources
    print("Available Resources:")
    resources = await server.resource_registry.list_resources()
    for resource in resources:
        print(f"- {resource.name}")
        print(f"  URI: {resource.uri}")
        print(f"  Type: {resource.mimeType}")
        print(f"  Description: {resource.description}")
        print()

    # Demonstrate resource content retrieval
    print("Resource Content Examples:")

    test_uris = [
        "memory://docs/mcp-protocol",
        "memory://guides/server-implementation",
        "memory://examples/server-config"
    ]

    for uri in test_uris:
        try:
            content = await server.resource_registry.get_resource_content(uri)
            preview = content.text[:200] + "..." if len(content.text) > 200 else content.text # noqa
            print(f"✅ {uri}:")
            print(f"   {preview}")
            print()
        except Exception as e:
            print(f"❌ {uri}: Error - {e}")
            print()

    print("This server would normally be started with:")
    print("python -m mcp_course.examples.resources_server_example --run")
    print("And would wait for MCP Client connections via stdio.")


async def create_and_run_server():
    """Create and run the actual server (for production use)."""
    config = ServerConfig(
        name="resources-demo-server",
        version="1.0.0",
        description="MCP Server demonstrating comprehensive resource management"
    )

    server = ResourcesEnabledMCPServer(config)
    await server.run()


async def demonstrate_resource_discovery():
    """Demonstrate resource discovery capabilities."""
    print("=== Resource Discovery Demonstration ===")
    print()

    registry = create_example_resources()

    # Demonstrate file discovery
    print("1. File Resource Discovery:")
    try:
        # Discover Python files in current directory
        current_dir_uri = f"file://{Path.cwd()}"
        discovered = await registry.discover_resources(current_dir_uri)

        python_files = [r for r in discovered if r.uri.endswith('.py')]
        print(f"   Found {len(python_files)} Python files:")
        for resource in python_files[:5]:  # Show first 5
            print(f"   - {resource.name} ({resource.uri})")

        if len(python_files) > 5: # noqa
            print(f"   ... and {len(python_files) - 5} more")

    except Exception as e:
        print(f"   Error: {e}")

    print()

    # Demonstrate metadata retrieval
    print("2. Resource Metadata:")
    try:
        metadata = await registry.get_resource_metadata("memory://sample/greeting")
        print(f"   URI: {metadata.uri}")
        print(f"   Name: {metadata.name}")
        print(f"   MIME Type: {metadata.mime_type}")
        print(f"   Size: {metadata.size} bytes")
        print(f"   Tags: {metadata.tags}")
    except Exception as e:
        print(f"   Error: {e}")

    print()


async def main():
    """Main entry point - demonstrate or run based on context."""

    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        # Actually run the server
        await create_and_run_server()
    elif len(sys.argv) > 1 and sys.argv[1] == "--discovery":
        # Demonstrate resource discovery
        await demonstrate_resource_discovery()
    else:
        # Just demonstrate the server
        await demonstrate_server()


if __name__ == "__main__":
    asyncio.run(main())
