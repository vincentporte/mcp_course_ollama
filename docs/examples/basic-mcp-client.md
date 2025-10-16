# Basic MCP Client

## Description
A simple MCP client that connects to servers and uses tools

## Learning Objectives
- Understand MCP client architecture
- Learn how to connect to MCP servers
- Understand tool discovery and execution
- Learn resource access patterns

## Prerequisites
- Basic Python knowledge
- Understanding of async/await
- Familiarity with MCP server concepts

## Key Concepts
- Client session management
- Server connection and initialization
- Tool discovery and execution
- Resource listing and reading
- Error handling in client code

## Difficulty Level
Beginner

## Code Example

```python
"""
Basic MCP Client Example

This example demonstrates how to create an MCP client that connects
to servers, lists available tools, and executes them.
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.models import InitializationOptions

class BasicMCPClient:
    """
    Basic MCP Client implementation
    
    This class demonstrates the fundamental patterns for connecting
    to MCP servers and using their capabilities.
    """
    
    def __init__(self, server_command: str, server_args: list = None):
        """
        Initialize the client with server connection parameters
        
        Args:
            server_command: Command to start the MCP server
            server_args: Arguments for the server command
        """
        self.server_params = StdioServerParameters(
            command=server_command,
            args=server_args or []
        )
        self.session = None
    
    async def connect(self):
        """
        Connect to the MCP server
        
        This method establishes a connection and initializes the session.
        """
        try:
            # Create client session
            self.session = ClientSession(self.server_params)
            
            # Initialize the connection
            await self.session.initialize()
            
            print("Successfully connected to MCP server")
            return True
            
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            return False
    
    async def list_available_tools(self):
        """
        List all tools available from the connected server
        
        Returns:
            List of available tools with their descriptions
        """
        if not self.session:
            raise RuntimeError("Not connected to server")
        
        try:
            # Request list of tools from server
            tools_response = await self.session.list_tools()
            
            print("Available tools:")
            for tool in tools_response.tools:
                print(f"  - {tool.name}: {tool.description}")
                
                # Show input schema if available
                if hasattr(tool, 'inputSchema') and tool.inputSchema:
                    print(f"    Parameters: {json.dumps(tool.inputSchema, indent=6)}")
            
            return tools_response.tools
            
        except Exception as e:
            print(f"Error listing tools: {e}")
            return []
    
    async def call_tool(self, tool_name: str, arguments: dict):
        """
        Call a specific tool with given arguments
        
        Args:
            tool_name: Name of the tool to call
            arguments: Dictionary of arguments for the tool
            
        Returns:
            Tool execution result
        """
        if not self.session:
            raise RuntimeError("Not connected to server")
        
        try:
            print(f"Calling tool '{tool_name}' with arguments: {arguments}")
            
            # Execute the tool
            result = await self.session.call_tool(tool_name, arguments)
            
            print(f"Tool result:")
            for content in result.content:
                if hasattr(content, 'text'):
                    print(f"  {content.text}")
                else:
                    print(f"  {content}")
            
            return result
            
        except Exception as e:
            print(f"Error calling tool '{tool_name}': {e}")
            return None
    
    async def list_resources(self):
        """
        List all resources available from the server
        
        Returns:
            List of available resources
        """
        if not self.session:
            raise RuntimeError("Not connected to server")
        
        try:
            resources_response = await self.session.list_resources()
            
            print("Available resources:")
            for resource in resources_response.resources:
                print(f"  - {resource.name} ({resource.uri})")
                if hasattr(resource, 'description'):
                    print(f"    {resource.description}")
            
            return resources_response.resources
            
        except Exception as e:
            print(f"Error listing resources: {e}")
            return []
    
    async def read_resource(self, uri: str):
        """
        Read content from a specific resource
        
        Args:
            uri: URI of the resource to read
            
        Returns:
            Resource content
        """
        if not self.session:
            raise RuntimeError("Not connected to server")
        
        try:
            print(f"Reading resource: {uri}")
            
            result = await self.session.read_resource(uri)
            
            print("Resource content:")
            for content in result.contents:
                if hasattr(content, 'text'):
                    print(content.text)
                else:
                    print(str(content))
            
            return result
            
        except Exception as e:
            print(f"Error reading resource '{uri}': {e}")
            return None
    
    async def disconnect(self):
        """
        Disconnect from the MCP server
        """
        if self.session:
            await self.session.close()
            self.session = None
            print("Disconnected from server")

async def demo_client_usage():
    """
    Demonstrate basic client usage patterns
    """
    # Create client instance
    client = BasicMCPClient("python", ["basic_server.py"])
    
    try:
        # Connect to server
        if await client.connect():
            
            # List available tools
            tools = await client.list_available_tools()
            
            # Try calling some tools
            if tools:
                # Call echo tool
                await client.call_tool("echo", {"message": "Hello, MCP!"})
                
                # Call add_numbers tool
                await client.call_tool("add_numbers", {"a": 5, "b": 3})
            
            # List and read resources
            resources = await client.list_resources()
            if resources:
                # Read the first resource
                await client.read_resource(resources[0].uri)
    
    finally:
        # Always disconnect
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(demo_client_usage())
```

## Code Annotations

**Line 20:** Client initialization with server parameters
```python
    
```

**Line 35:** Connection establishment and session initialization
```python
    async def connect(self):
```

**Line 55:** Tool discovery - getting available tools from server
```python
    async def list_available_tools(self):
```

**Line 82:** Tool execution with parameter validation
```python
    
```

**Line 115:** Resource discovery and listing
```python
    
```

**Line 140:** Resource content reading
```python
    
```

**Line 165:** Proper cleanup and disconnection
```python
            
```

## Dependencies
- mcp

