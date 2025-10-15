"""Basic MCP Client implementation with connection and communication patterns."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import (
    CallToolResult,
    ListResourcesResult,
    ListToolsResult,
    ReadResourceResult,
    Resource,
    Tool,
)


@dataclass
class ClientConfig:
    """Configuration for MCP Client."""
    name: str = "mcp-course-client"
    version: str = "1.0.0"
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    logging_level: str = "INFO"


@dataclass
class ServerConnection:
    """Represents a connection to an MCP Server."""
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    session: ClientSession | None = None
    connected: bool = False
    last_error: str | None = None
    connection_time: datetime | None = None


class BasicMCPClient:
    """
    A basic MCP Client implementation demonstrating core communication patterns.

    This class shows how to:
    - Connect to MCP Servers using stdio transport
    - Discover server capabilities (tools, resources, prompts)
    - Execute tools with proper error handling
    - Manage multiple server connections
    - Handle connection failures and recovery
    """

    def __init__(self, config: ClientConfig = None):
        """Initialize the basic MCP Client."""
        self.config = config or ClientConfig()
        self.logger = logging.getLogger(self.config.name)
        self.servers: dict[str, ServerConnection] = {}
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging for the client."""
        logging.basicConfig(
            level=getattr(logging, self.config.logging_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger.info(f"Initialized {self.config.name} v{self.config.version}")

    async def add_server(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str]  | None = None
    ) -> bool:
        """
        Add a server configuration for connection.

        Args:
            name: Unique name for the server
            command: Command to start the server
            args: Command line arguments
            env: Environment variables

        Returns:
            True if server was added successfully
        """
        if name in self.servers:
            self.logger.warning(f"Server {name} already exists, updating configuration")

        self.servers[name] = ServerConnection(
            name=name,
            command=command,
            args=args or [],
            env=env or {}
        )

        self.logger.info(f"Added server configuration: {name}")
        return True

    async def connect_to_server(self, server_name: str) -> bool:
        """
        Connect to a specific MCP Server.

        Args:
            server_name: Name of the server to connect to

        Returns:
            True if connection was successful
        """
        if server_name not in self.servers:
            self.logger.error(f"Server {server_name} not found in configuration")
            return False

        server = self.servers[server_name]

        try:
            self.logger.info(f"Connecting to server: {server_name}")

            # Create server parameters
            server_params = StdioServerParameters(
                command=server.command,
                args=server.args,
                env=server.env
            )

            # Create client session
            session = await stdio_client(server_params)

            # Initialize the session
            await session.initialize()

            # Update server connection info
            server.session = session
            server.connected = True
            server.connection_time = datetime.now()
            server.last_error = None

            self.logger.info(f"Successfully connected to server: {server_name}")
            return True

        except Exception as e:
            error_msg = f"Failed to connect to server {server_name}: {e}"
            self.logger.error(error_msg)
            server.last_error = error_msg
            server.connected = False
            return False

    async def disconnect_from_server(self, server_name: str) -> bool:
        """
        Disconnect from a specific MCP Server.

        Args:
            server_name: Name of the server to disconnect from

        Returns:
            True if disconnection was successful
        """
        if server_name not in self.servers:
            self.logger.error(f"Server {server_name} not found")
            return False

        server = self.servers[server_name]

        try:
            if server.session and server.connected:
                await server.session.close()
                self.logger.info(f"Disconnected from server: {server_name}")

            server.session = None
            server.connected = False
            return True

        except Exception as e:
            self.logger.error(f"Error disconnecting from server {server_name}: {e}")
            return False

    async def list_server_tools(self, server_name: str) -> list[Tool] | None:
        """
        List available tools from a specific server.

        Args:
            server_name: Name of the server to query

        Returns:
            List of available tools or None if error
        """
        server = self._get_connected_server(server_name)
        if not server:
            return None

        try:
            self.logger.info(f"Listing tools from server: {server_name}")

            result: ListToolsResult = await server.session.list_tools()
            tools = result.tools

            self.logger.info(f"Found {len(tools)} tools on server {server_name}")
            for tool in tools:
                self.logger.debug(f"  - {tool.name}: {tool.description}")

            return tools

        except Exception as e:
            self.logger.error(f"Error listing tools from server {server_name}: {e}")
            return None

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any]
    ) -> CallToolResult | None:
        """
        Call a tool on a specific server.

        Args:
            server_name: Name of the server
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool execution result or None if error
        """
        server = self._get_connected_server(server_name)
        if not server:
            return None

        try:
            self.logger.info(f"Calling tool {tool_name} on server {server_name}")
            self.logger.debug(f"Tool arguments: {arguments}")

            result: CallToolResult = await server.session.call_tool(tool_name, arguments)

            self.logger.info(f"Tool {tool_name} executed successfully")
            self.logger.debug(f"Tool result: {result}")

            return result

        except Exception as e:
            self.logger.error(f"Error calling tool {tool_name} on server {server_name}: {e}")
            return None

    async def list_server_resources(self, server_name: str) -> list[Resource] | None:
        """
        List available resources from a specific server.

        Args:
            server_name: Name of the server to query

        Returns:
            List of available resources or None if error
        """
        server = self._get_connected_server(server_name)
        if not server:
            return None

        try:
            self.logger.info(f"Listing resources from server: {server_name}")

            result: ListResourcesResult = await server.session.list_resources()
            resources = result.resources

            self.logger.info(f"Found {len(resources)} resources on server {server_name}")
            for resource in resources:
                self.logger.debug(f"  - {resource.uri}: {resource.name}")

            return resources

        except Exception as e:
            self.logger.error(f"Error listing resources from server {server_name}: {e}")
            return None

    async def read_resource(
        self,
        server_name: str,
        resource_uri: str
    ) -> ReadResourceResult | None:
        """
        Read a resource from a specific server.

        Args:
            server_name: Name of the server
            resource_uri: URI of the resource to read

        Returns:
            Resource content or None if error
        """
        server = self._get_connected_server(server_name)
        if not server:
            return None

        try:
            self.logger.info(f"Reading resource {resource_uri} from server {server_name}")

            result: ReadResourceResult = await server.session.read_resource(resource_uri)

            self.logger.info(f"Resource {resource_uri} read successfully")
            return result

        except Exception as e:
            self.logger.error(f"Error reading resource {resource_uri} from server {server_name}: {e}")
            return None

    async def get_server_status(self, server_name: str) -> dict[str, Any]:
        """
        Get status information for a server connection.

        Args:
            server_name: Name of the server

        Returns:
            Dictionary containing server status information
        """
        if server_name not in self.servers:
            return {"error": f"Server {server_name} not found"}

        server = self.servers[server_name]

        status = {
            "name": server.name,
            "connected": server.connected,
            "command": server.command,
            "args": server.args,
            "connection_time": server.connection_time.isoformat() if server.connection_time else None,
            "last_error": server.last_error
        }

        if server.connected and server.session:
            try:
                # Try to get server capabilities
                tools = await self.list_server_tools(server_name)
                resources = await self.list_server_resources(server_name)

                status.update({
                    "tools_count": len(tools) if tools else 0,
                    "resources_count": len(resources) if resources else 0,
                    "capabilities": {
                        "tools": tools is not None,
                        "resources": resources is not None
                    }
                })
            except Exception as e:
                status["capability_check_error"] = str(e)

        return status

    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        self.logger.info("Disconnecting from all servers")

        for server_name in list(self.servers.keys()):
            await self.disconnect_from_server(server_name)

        self.logger.info("Disconnected from all servers")

    def _get_connected_server(self, server_name: str) -> ServerConnection | None:
        """
        Get a connected server or log appropriate error.

        Args:
            server_name: Name of the server

        Returns:
            ServerConnection if connected, None otherwise
        """
        if server_name not in self.servers:
            self.logger.error(f"Server {server_name} not found in configuration")
            return None

        server = self.servers[server_name]

        if not server.connected or not server.session:
            self.logger.error(f"Server {server_name} is not connected")
            return None

        return server

    def get_connected_servers(self) -> list[str]:
        """
        Get list of currently connected server names.

        Returns:
            List of connected server names
        """
        return [name for name, server in self.servers.items() if server.connected]

    def get_all_servers(self) -> list[str]:
        """
        Get list of all configured server names.

        Returns:
            List of all server names
        """
        return list(self.servers.keys())


async def demonstrate_basic_client_usage():
    """
    Demonstrate basic MCP Client usage patterns.

    This function shows a complete example of:
    - Creating a client
    - Adding server configurations
    - Connecting to servers
    - Discovering capabilities
    - Calling tools
    - Handling errors
    """
    # Create client
    client = BasicMCPClient()

    try:
        # Add a server configuration (example - would need actual server)
        await client.add_server(
            name="tutorial-server",
            command="python",
            args=["-m", "mcp_course.server.basic"]
        )

        # Connect to the server
        connected = await client.connect_to_server("tutorial-server")
        if not connected:
            print("Failed to connect to server")
            return

        # List available tools
        tools = await client.list_server_tools("tutorial-server")
        if tools:
            print(f"Available tools: {[tool.name for tool in tools]}")

            # Call a tool
            result = await client.call_tool(
                "tutorial-server",
                "echo",
                {"message": "Hello from MCP Client!"}
            )

            if result:
                print(f"Tool result: {result.content}")

        # Get server status
        status = await client.get_server_status("tutorial-server")
        print(f"Server status: {status}")

    except Exception as e:
        print(f"Error in demonstration: {e}")

    finally:
        # Clean up
        await client.disconnect_all()


if __name__ == "__main__":
    # Run the demonstration when executed directly
    asyncio.run(demonstrate_basic_client_usage())
