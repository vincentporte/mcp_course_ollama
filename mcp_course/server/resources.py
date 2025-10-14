"""
MCP Resources Exposure and Management System

This module provides utilities for creating, managing, and exposing MCP resources
with proper URI handling, content providers, and metadata systems.
"""

from abc import ABC, abstractmethod
import base64
from dataclasses import dataclass, field
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from mcp.types import EmbeddedResource, ImageContent, Resource, TextContent


@dataclass
class ResourceMetadata:
    """Metadata for an MCP resource."""

    uri: str
    name: str
    description: str
    mime_type: str
    size: int | None = None
    last_modified: str | None = None
    tags: list[str] = field(default_factory=list)
    custom_attributes: dict[str, Any] = field(default_factory=dict)

    def to_mcp_resource(self) -> Resource:
        """Convert to MCP Resource format."""
        return Resource(
            uri=self.uri,
            name=self.name,
            description=self.description,
            mimeType=self.mime_type
        )


class ResourceProvider(ABC):
    """
    Abstract base class for resource content providers.

    Resource providers handle the actual retrieval and formatting
    of resource content from various sources (files, databases, APIs, etc.).
    """

    @abstractmethod
    async def can_handle(self, uri: str) -> bool:
        """Check if this provider can handle the given URI."""
        pass

    @abstractmethod
    async def get_content(self, uri: str) -> TextContent | ImageContent | EmbeddedResource:
        """Retrieve the content for the given URI."""
        pass

    @abstractmethod
    async def get_metadata(self, uri: str) -> ResourceMetadata:
        """Get metadata for the given URI."""
        pass


class FileResourceProvider(ResourceProvider):
    """
    Resource provider for local file system resources.

    Handles file:// URIs and provides content from local files
    with appropriate MIME type detection.
    """

    def __init__(self, base_path: Path | None = None, allowed_extensions: list[str] | None = None):
        """
        Initialize the file resource provider.

        Args:
            base_path: Base directory for file access (for security)
            allowed_extensions: List of allowed file extensions
        """
        self.base_path = base_path or Path.cwd()
        self.allowed_extensions = allowed_extensions or [
            '.txt', '.md', '.json', '.yaml', '.yml', '.csv', '.xml', '.html'
        ]

    async def can_handle(self, uri: str) -> bool:
        """Check if this is a file URI we can handle."""
        parsed = urlparse(uri)
        if parsed.scheme != 'file':
            return False

        path = Path(parsed.path)
        return path.suffix.lower() in self.allowed_extensions

    async def get_content(self, uri: str) -> TextContent | ImageContent:
        """Get file content."""
        parsed = urlparse(uri)
        file_path = self.base_path / Path(parsed.path).relative_to('/')

        # Security check - ensure path is within base_path
        try:
            file_path.resolve().relative_to(self.base_path.resolve())
        except ValueError as e:
            raise ValueError(f"Access denied: {uri} is outside allowed directory") from e

        if not file_path.exists():
            raise FileNotFoundError(f"Resource not found: {uri}")

        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        mime_type = mime_type or 'text/plain'

        # Read content based on MIME type
        if mime_type.startswith('text/') or mime_type in ['application/json', 'application/xml']:
            content = file_path.read_text(encoding='utf-8')
            return TextContent(type="text", text=content)
        else:
            # For binary files, return as embedded resource
            content = file_path.read_bytes()
            encoded_content = base64.b64encode(content).decode('utf-8')
            return EmbeddedResource(
                type="resource",
                resource=Resource(
                    uri=uri,
                    name=file_path.name,
                    mimeType=mime_type
                ),
                text=encoded_content
            )

    async def get_metadata(self, uri: str) -> ResourceMetadata:
        """Get file metadata."""
        parsed = urlparse(uri)
        file_path = self.base_path / Path(parsed.path).relative_to('/')

        if not file_path.exists():
            raise FileNotFoundError(f"Resource not found: {uri}")

        stat = file_path.stat()
        mime_type, _ = mimetypes.guess_type(str(file_path))

        return ResourceMetadata(
            uri=uri,
            name=file_path.name,
            description=f"File: {file_path.name}",
            mime_type=mime_type or 'application/octet-stream',
            size=stat.st_size,
            last_modified=str(stat.st_mtime),
            tags=['file', file_path.suffix[1:] if file_path.suffix else 'no-extension']
        )


class MemoryResourceProvider(ResourceProvider):
    """
    Resource provider for in-memory resources.

    Useful for dynamic content, computed data, or cached resources.
    """

    def __init__(self):
        """Initialize the memory resource provider."""
        self.resources: dict[str, dict[str, Any]] = {}

    def add_resource(
        self,
        uri: str,
        content: str,
        name: str,
        description: str,
        mime_type: str = "text/plain",
        metadata: dict[str, Any] | None = None
    ):
        """Add a resource to memory storage."""
        self.resources[uri] = {
            'content': content,
            'metadata': ResourceMetadata(
                uri=uri,
                name=name,
                description=description,
                mime_type=mime_type,
                size=len(content.encode('utf-8')),
                custom_attributes=metadata or {}
            )
        }

    async def can_handle(self, uri: str) -> bool:
        """Check if this URI is in memory storage."""
        return uri in self.resources

    async def get_content(self, uri: str) -> TextContent:
        """Get content from memory."""
        if uri not in self.resources:
            raise ValueError(f"Resource not found: {uri}")

        content = self.resources[uri]['content']
        return TextContent(type="text", text=content)

    async def get_metadata(self, uri: str) -> ResourceMetadata:
        """Get metadata from memory."""
        if uri not in self.resources:
            raise ValueError(f"Resource not found: {uri}")

        return self.resources[uri]['metadata']


class APIResourceProvider(ResourceProvider):
    """
    Resource provider for API-based resources.

    Handles http:// and https:// URIs by making HTTP requests
    to external APIs and formatting the responses.
    """

    def __init__(self, allowed_hosts: list[str] | None = None):
        """
        Initialize the API resource provider.

        Args:
            allowed_hosts: List of allowed hostnames for security
        """
        self.allowed_hosts = allowed_hosts

    async def can_handle(self, uri: str) -> bool:
        """Check if this is an HTTP URI we can handle."""
        parsed = urlparse(uri)
        if parsed.scheme not in ['http', 'https']:
            return False

        return not (self.allowed_hosts and parsed.hostname not in self.allowed_hosts)

    async def get_content(self, uri: str) -> TextContent:
        """Get content from API endpoint."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(uri, timeout=30.0)
                response.raise_for_status()

                # Try to format JSON responses nicely
                content_type = response.headers.get('content-type', '')
                if 'application/json' in content_type:
                    try:
                        json_data = response.json()
                        formatted_content = json.dumps(json_data, indent=2)
                    except json.JSONDecodeError:
                        formatted_content = response.text
                else:
                    formatted_content = response.text

                return TextContent(type="text", text=formatted_content)

        except Exception as e:
            raise ValueError(f"Failed to fetch resource {uri}: {e}") from e

    async def get_metadata(self, uri: str) -> ResourceMetadata:
        """Get metadata from API endpoint."""
        try:
            async with httpx.AsyncClient() as client:
                # Use HEAD request to get metadata without full content
                response = await client.head(uri, timeout=10.0)
                response.raise_for_status()

                content_type = response.headers.get('content-type', 'text/plain')
                content_length = response.headers.get('content-length')

                return ResourceMetadata(
                    uri=uri,
                    name=urlparse(uri).path.split('/')[-1] or 'api-resource',
                    description=f"API resource: {uri}",
                    mime_type=content_type.split(';')[0],  # Remove charset info
                    size=int(content_length) if content_length else None,
                    tags=['api', 'http']
                )

        except Exception as e:
            raise ValueError(f"Failed to get metadata for {uri}: {e}") from e


class ResourceRegistry:
    """
    Registry for managing MCP resources and their providers.

    This class provides a centralized way to register resource providers,
    discover available resources, and handle resource access requests.
    """

    def __init__(self):
        """Initialize the resource registry."""
        self.providers: list[ResourceProvider] = []
        self.static_resources: dict[str, ResourceMetadata] = {}

    def add_provider(self, provider: ResourceProvider):
        """Add a resource provider to the registry."""
        self.providers.append(provider)

    def add_static_resource(self, metadata: ResourceMetadata):
        """Add a static resource definition."""
        self.static_resources[metadata.uri] = metadata

    async def list_resources(self, uri_pattern: str | None = None) -> list[Resource]:
        """
        List all available resources.

        Args:
            uri_pattern: Optional pattern to filter resources

        Returns:
            List of MCP Resource objects
        """
        resources = []

        # Add static resources
        for metadata in self.static_resources.values():
            if not uri_pattern or uri_pattern in metadata.uri:
                resources.append(metadata.to_mcp_resource())

        return resources

    async def get_resource_content(self, uri: str) -> TextContent | ImageContent | EmbeddedResource:
        """
        Get content for a specific resource URI.

        Args:
            uri: Resource URI to retrieve

        Returns:
            Resource content in appropriate format

        Raises:
            ValueError: If no provider can handle the URI
        """
        # Find a provider that can handle this URI
        for provider in self.providers:
            if await provider.can_handle(uri):
                return await provider.get_content(uri)

        raise ValueError(f"No provider available for resource: {uri}")

    async def get_resource_metadata(self, uri: str) -> ResourceMetadata:
        """
        Get metadata for a specific resource URI.

        Args:
            uri: Resource URI

        Returns:
            Resource metadata

        Raises:
            ValueError: If no provider can handle the URI
        """
        # Check static resources first
        if uri in self.static_resources:
            return self.static_resources[uri]

        # Find a provider that can handle this URI
        for provider in self.providers:
            if await provider.can_handle(uri):
                return await provider.get_metadata(uri)

        raise ValueError(f"No provider available for resource: {uri}")

    async def discover_resources(self, base_uri: str) -> list[Resource]:
        """
        Discover resources under a base URI.

        This method attempts to find all available resources
        under a given base URI (like a directory listing).

        Args:
            base_uri: Base URI to search under

        Returns:
            List of discovered resources
        """
        discovered = []

        # For file URIs, list directory contents
        if base_uri.startswith('file://'):
            parsed = urlparse(base_uri)
            base_path = Path(parsed.path)

            if base_path.is_dir():
                for file_path in base_path.rglob('*'):
                    if file_path.is_file():
                        file_uri = f"file://{file_path}"
                        try:
                            metadata = await self.get_resource_metadata(file_uri)
                            discovered.append(metadata.to_mcp_resource())
                        except Exception:
                            continue  # Skip files we can't handle

        return discovered


def create_example_resources() -> ResourceRegistry:
    """
    Create a resource registry with example resources.

    This function demonstrates how to set up various types of resources
    for educational purposes.

    Returns:
        Configured ResourceRegistry with example resources
    """
    registry = ResourceRegistry()

    # Add file provider
    file_provider = FileResourceProvider(
        base_path=Path.cwd(),
        allowed_extensions=['.txt', '.md', '.json', '.py', '.yaml']
    )
    registry.add_provider(file_provider)

    # Add memory provider with sample data
    memory_provider = MemoryResourceProvider()

    # Add some sample in-memory resources
    memory_provider.add_resource(
        uri="memory://sample/greeting",
        content="Hello from MCP Resources!",
        name="Sample Greeting",
        description="A simple greeting message stored in memory",
        mime_type="text/plain"
    )

    memory_provider.add_resource(
        uri="memory://sample/config",
        content=json.dumps({
            "server_name": "example-server",
            "version": "1.0.0",
            "features": ["tools", "resources", "prompts"]
        }, indent=2),
        name="Sample Configuration",
        description="Example server configuration in JSON format",
        mime_type="application/json"
    )

    registry.add_provider(memory_provider)

    # Add API provider (with restricted hosts for security)
    api_provider = APIResourceProvider(
        allowed_hosts=['httpbin.org', 'jsonplaceholder.typicode.com']
    )
    registry.add_provider(api_provider)

    # Add some static resource definitions
    registry.add_static_resource(ResourceMetadata(
        uri="memory://sample/greeting",
        name="Sample Greeting",
        description="A simple greeting message",
        mime_type="text/plain",
        tags=["example", "greeting"]
    ))

    registry.add_static_resource(ResourceMetadata(
        uri="memory://sample/config",
        name="Sample Configuration",
        description="Example server configuration",
        mime_type="application/json",
        tags=["example", "config"]
    ))

    return registry


async def demonstrate_resources():
    """
    Demonstrate the resource management system.

    This function shows how to use the resource registry and providers
    to manage different types of resources.
    """
    print("=== MCP Resources Management Demonstration ===")
    print()

    # Create registry with examples
    registry = create_example_resources()

    print("1. Listing Available Resources:")
    resources = await registry.list_resources()
    for resource in resources:
        print(f"   - {resource.name} ({resource.uri})")
        print(f"     Type: {resource.mimeType}")
        print(f"     Description: {resource.description}")
        print()

    print("2. Getting Resource Content:")

    # Test memory resources
    try:
        content = await registry.get_resource_content("memory://sample/greeting")
        print(f"   Greeting: {content.text}")
    except Exception as e:
        print(f"   Error: {e}")

    try:
        content = await registry.get_resource_content("memory://sample/config")
        print(f"   Config: {content.text[:100]}...")
    except Exception as e:
        print(f"   Error: {e}")

    print()

    print("3. Resource Metadata:")
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
    print("Resource management demonstration complete!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_resources())
