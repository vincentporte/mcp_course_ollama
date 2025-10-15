"""Connection management and server discovery for MCP Clients."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from mcp_course.client.basic import BasicMCPClient, ServerConnection


@dataclass
class ServerRegistry:
    """Registry of known MCP Servers."""
    servers: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_updated: datetime | None = None

    def add_server(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str]  | None = None,
        description: str = "",
        tags: list[str] | None = None
    ) -> None:
        """Add a server to the registry."""
        self.servers[name] = {
            "command": command,
            "args": args or [],
            "env": env or {},
            "description": description,
            "tags": tags or [],
            "added_at": datetime.now().isoformat()
        }
        self.last_updated = datetime.now()

    def remove_server(self, name: str) -> bool:
        """Remove a server from the registry."""
        if name in self.servers:
            del self.servers[name]
            self.last_updated = datetime.now()
            return True
        return False

    def get_server(self, name: str) -> dict[str, Any] | None:
        """Get server configuration by name."""
        return self.servers.get(name)

    def list_servers(self, tag: str | None = None) -> list[str]:
        """List server names, optionally filtered by tag."""
        if tag is None:
            return list(self.servers.keys())

        return [
            name for name, config in self.servers.items()
            if tag in config.get("tags", [])
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert registry to dictionary for serialization."""
        return {
            "servers": self.servers,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ServerRegistry":
        """Create registry from dictionary."""
        registry = cls()
        registry.servers = data.get("servers", {})
        last_updated_str = data.get("last_updated")
        if last_updated_str:
            registry.last_updated = datetime.fromisoformat(last_updated_str)
        return registry


class ServerDiscovery:
    """
    Server discovery and management system.

    This class provides functionality to:
    - Discover MCP Servers in the system
    - Maintain a registry of known servers
    - Validate server availability
    - Load server configurations from files
    """

    def __init__(self, registry_path: Path | None = None):
        """Initialize server discovery."""
        self.registry_path = registry_path or Path.home() / ".mcp" / "servers.json"
        self.registry = ServerRegistry()
        self.logger = logging.getLogger("ServerDiscovery")
        self._load_registry()

    def _load_registry(self) -> None:
        """Load server registry from file."""
        try:
            if self.registry_path.exists():
                with Path.open(self.registry_path) as f:
                    data = json.load(f)
                self.registry = ServerRegistry.from_dict(data)
                self.logger.info(f"Loaded {len(self.registry.servers)} servers from registry")
            else:
                self.logger.info("No existing registry found, starting with empty registry")
        except Exception as e:
            self.logger.error(f"Error loading registry: {e}")
            self.registry = ServerRegistry()

    def _save_registry(self) -> None:
        """Save server registry to file."""
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            with Path.open(self.registry_path, 'w') as f:
                json.dump(self.registry.to_dict(), f, indent=2)
            self.logger.info("Registry saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving registry: {e}")

    def add_server(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str]  | None = None,
        description: str = "",
        tags: list[str] | None = None,
        save: bool = True
    ) -> bool:
        """
        Add a server to the discovery registry.

        Args:
            name: Unique server name
            command: Command to start the server
            args: Command arguments
            env: Environment variables
            description: Server description
            tags: Tags for categorization
            save: Whether to save registry to disk

        Returns:
            True if server was added successfully
        """
        try:
            self.registry.add_server(name, command, args, env, description, tags)
            if save:
                self._save_registry()
            self.logger.info(f"Added server to registry: {name}")
            return True
        except Exception as e:
            self.logger.error(f"Error adding server {name}: {e}")
            return False

    def remove_server(self, name: str, save: bool = True) -> bool:
        """
        Remove a server from the registry.

        Args:
            name: Server name to remove
            save: Whether to save registry to disk

        Returns:
            True if server was removed successfully
        """
        try:
            if self.registry.remove_server(name):
                if save:
                    self._save_registry()
                self.logger.info(f"Removed server from registry: {name}")
                return True
            else:
                self.logger.warning(f"Server not found in registry: {name}")
                return False
        except Exception as e:
            self.logger.error(f"Error removing server {name}: {e}")
            return False

    def discover_system_servers(self) -> list[dict[str, Any]]:
        """
        Discover MCP Servers available in the system.

        This method looks for common MCP Server patterns:
        - Python packages with MCP server entry points
        - Executable files with MCP in the name
        - Known MCP server locations

        Returns:
            List of discovered server configurations
        """
        discovered = []

        # Look for Python MCP packages
        try:
            result = subprocess.run(
                ["pip", "list"],
                check=False, capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'mcp' in line.lower() and 'server' in line.lower():
                        package_name = line.split()[0]
                        discovered.append({
                            "name": f"python-{package_name}",
                            "command": "python",
                            "args": ["-m", package_name],
                            "description": f"Python MCP server package: {package_name}",
                            "tags": ["python", "pip", "discovered"]
                        })
        except Exception as e:
            self.logger.debug(f"Error discovering Python packages: {e}")

        # Look for MCP executables in PATH
        try:
            path_dirs = [Path(p) for p in os.environ.get("PATH", "").split(os.pathsep)]
            for path_dir in path_dirs:
                if path_dir.exists():
                    for executable in path_dir.glob("*mcp*"):
                        if executable.is_file() and os.access(executable, os.X_OK):
                            discovered.append({
                                "name": f"executable-{executable.stem}",
                                "command": str(executable),
                                "args": [],
                                "description": f"MCP executable: {executable.name}",
                                "tags": ["executable", "discovered"]
                            })
        except Exception as e:
            self.logger.debug(f"Error discovering executables: {e}")

        self.logger.info(f"Discovered {len(discovered)} potential MCP servers")
        return discovered

    async def validate_server(self, name: str) -> dict[str, Any]:
        """
        Validate that a server can be started and responds correctly.

        Args:
            name: Server name to validate

        Returns:
            Dictionary containing validation results
        """
        server_config = self.registry.get_server(name)
        if not server_config:
            return {"valid": False, "error": f"Server {name} not found in registry"}

        validation_result = {
            "valid": False,
            "server_name": name,
            "command_exists": False,
            "can_connect": False,
            "tools_available": False,
            "error": None,
            "validation_time": datetime.now().isoformat()
        }

        try:
            # Check if command exists
            command = server_config["command"]
            if shutil.which(command):
                validation_result["command_exists"] = True
            else:
                validation_result["error"] = f"Command not found: {command}"
                return validation_result

            # Try to connect with a timeout
            client = BasicMCPClient()
            await client.add_server(
                name=f"validation-{name}",
                command=command,
                args=server_config.get("args", []),
                env=server_config.get("env", {})
            )

            # Attempt connection with timeout
            connected = await asyncio.wait_for(
                client.connect_to_server(f"validation-{name}"),
                timeout=10.0
            )

            if connected:
                validation_result["can_connect"] = True

                # Try to list tools
                tools = await asyncio.wait_for(
                    client.list_server_tools(f"validation-{name}"),
                    timeout=5.0
                )

                if tools is not None:
                    validation_result["tools_available"] = True
                    validation_result["tools_count"] = len(tools)

                validation_result["valid"] = True
            else:
                validation_result["error"] = "Failed to connect to server"

            # Clean up
            await client.disconnect_all()

        except TimeoutError:
            validation_result["error"] = "Server validation timed out"
        except Exception as e:
            validation_result["error"] = f"Validation error: {e}"

        return validation_result

    def load_servers_from_config(self, config_path: Path) -> int:
        """
        Load server configurations from a JSON config file.

        Args:
            config_path: Path to configuration file

        Returns:
            Number of servers loaded
        """
        try:
            with Path.open(config_path) as f:
                config = json.load(f)

            servers_loaded = 0
            servers_config = config.get("servers", {})

            for name, server_config in servers_config.items():
                self.add_server(
                    name=name,
                    command=server_config["command"],
                    args=server_config.get("args", []),
                    env=server_config.get("env", {}),
                    description=server_config.get("description", ""),
                    tags=server_config.get("tags", []),
                    save=False  # Save once at the end
                )
                servers_loaded += 1

            self._save_registry()
            self.logger.info(f"Loaded {servers_loaded} servers from config file")
            return servers_loaded

        except Exception as e:
            self.logger.error(f"Error loading servers from config: {e}")
            return 0

    def export_registry(self, export_path: Path) -> bool:
        """
        Export the current registry to a file.

        Args:
            export_path: Path to export file

        Returns:
            True if export was successful
        """
        try:
            with Path.open(export_path, 'w') as f:
                json.dump(self.registry.to_dict(), f, indent=2)
            self.logger.info(f"Registry exported to {export_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting registry: {e}")
            return False


class ConnectionManager:
    """
    Advanced connection management for MCP Clients.

    This class provides:
    - Connection pooling and reuse
    - Automatic reconnection with backoff
    - Health monitoring
    - Load balancing across multiple servers
    """

    def __init__(self, client: BasicMCPClient, max_connections: int = 10):
        """Initialize connection manager."""
        self.client = client
        self.max_connections = max_connections
        self.connection_pool: dict[str, ServerConnection] = {}
        self.health_check_interval = 30.0  # seconds
        self.reconnect_attempts = 3
        self.reconnect_delay = 5.0  # seconds
        self.logger = logging.getLogger("ConnectionManager")
        self._health_check_task: asyncio.Task | None = None

    async def start_health_monitoring(self) -> None:
        """Start background health monitoring."""
        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            self.logger.info("Started connection health monitoring")

    async def stop_health_monitoring(self) -> None:
        """Stop background health monitoring."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                self.logger("cancelledConnection")
            else:
                self.logger.error("Unexpected error stopping health check task")
            self._health_check_task = None
            self.logger.info("Stopped connection health monitoring")

    async def _health_check_loop(self) -> None:
        """Background loop for health checking connections."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_all_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")

    async def _check_all_connections(self) -> None:
        """Check health of all connections."""
        for server_name in self.client.get_connected_servers():
            try:
                # Simple health check - try to list tools
                tools = await self.client.list_server_tools(server_name)
                if tools is None:
                    self.logger.warning(f"Health check failed for server: {server_name}")
                    await self._attempt_reconnect(server_name)
            except Exception as e:
                self.logger.error(f"Health check error for server {server_name}: {e}")
                await self._attempt_reconnect(server_name)

    async def _attempt_reconnect(self, server_name: str) -> bool:
        """
        Attempt to reconnect to a server with exponential backoff.

        Args:
            server_name: Name of server to reconnect

        Returns:
            True if reconnection was successful
        """
        self.logger.info(f"Attempting to reconnect to server: {server_name}")

        for attempt in range(self.reconnect_attempts):
            try:
                # Disconnect first
                await self.client.disconnect_from_server(server_name)

                # Wait with exponential backoff
                delay = self.reconnect_delay * (2 ** attempt)
                await asyncio.sleep(delay)

                # Attempt reconnection
                if await self.client.connect_to_server(server_name):
                    self.logger.info(f"Successfully reconnected to server: {server_name}")
                    return True

            except Exception as e:
                self.logger.error(f"Reconnection attempt {attempt + 1} failed for {server_name}: {e}")

        self.logger.error(f"Failed to reconnect to server after {self.reconnect_attempts} attempts: {server_name}")
        return False

    async def get_healthy_servers(self) -> list[str]:
        """
        Get list of currently healthy server connections.

        Returns:
            List of healthy server names
        """
        healthy_servers = []

        for server_name in self.client.get_connected_servers():
            try:
                # Quick health check
                status = await self.client.get_server_status(server_name)
                if status.get("connected", False):
                    healthy_servers.append(server_name)
            except Exception as e:
                self.logger.debug(f"Server {server_name} failed health check: {e}")

        return healthy_servers

    async def shutdown(self) -> None:
        """Shutdown connection manager and clean up resources."""
        await self.stop_health_monitoring()
        await self.client.disconnect_all()
        self.logger.info("Connection manager shutdown complete")



async def demonstrate_connection_management():
    """Demonstrate connection management and server discovery."""
    # Create discovery system
    discovery = ServerDiscovery()

    # Add some example servers
    discovery.add_server(
        name="example-server",
        command="python",
        args=["-m", "mcp_course.server.basic"],
        description="Example MCP server for tutorials",
        tags=["tutorial", "example"]
    )

    # Discover system servers
    discovered = discovery.discover_system_servers()
    print(f"Discovered {len(discovered)} potential servers")

    # Create client and connection manager
    client = BasicMCPClient()
    conn_manager = ConnectionManager(client)

    try:
        # Start health monitoring
        await conn_manager.start_health_monitoring()

        # Add and connect to servers from registry
        for server_name in discovery.registry.list_servers():
            server_config = discovery.registry.get_server(server_name)
            if server_config:
                await client.add_server(
                    name=server_name,
                    command=server_config["command"],
                    args=server_config["args"],
                    env=server_config["env"]
                )

        # Get healthy servers
        healthy = await conn_manager.get_healthy_servers()
        print(f"Healthy servers: {healthy}")

    finally:
        # Clean up
        await conn_manager.shutdown()


if __name__ == "__main__":
    asyncio.run(demonstrate_connection_management())
