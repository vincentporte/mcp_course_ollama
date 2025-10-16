# Basic MCP Server

## Description
A minimal MCP server implementation demonstrating core concepts

## Learning Objectives
- Understand basic MCP server structure
- Learn how to define and implement tools
- Understand resource handling
- Learn proper error handling patterns

## Prerequisites
- Basic Python knowledge
- Understanding of async/await
- Familiarity with JSON

## Key Concepts
- MCP Server initialization
- Tool definition and implementation
- Resource exposure
- JSON-RPC message handling
- Error handling and validation

## Difficulty Level
Beginner

## Code Example

```python
"""
Basic MCP Server Example

This example demonstrates the fundamental structure of an MCP server
with a simple tool and resource implementation.
"""

import asyncio
import json
from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Create server instance
app = Server("basic-example-server")

@app.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools
    
    This handler returns the tools that this server provides.
    Each tool must have a name, description, and input schema.
    """
    return [
        types.Tool(
            name="echo",
            description="Echo back the input message",
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
        types.Tool(
            name="add_numbers", 
            description="Add two numbers together",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        )
    ]

@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """
    Handle tool execution
    
    This handler processes tool calls and returns results.
    It includes proper error handling and validation.
    """
    try:
        if name == "echo":
            message = arguments.get("message", "")
            return [
                types.TextContent(
                    type="text",
                    text=f"Echo: {message}"
                )
            ]
        
        elif name == "add_numbers":
            a = arguments.get("a")
            b = arguments.get("b")
            
            if a is None or b is None:
                raise ValueError("Both 'a' and 'b' parameters are required")
            
            result = a + b
            return [
                types.TextContent(
                    type="text", 
                    text=f"The sum of {a} and {b} is {result}"
                )
            ]
        
        else:
            raise ValueError(f"Unknown tool: {name}")
            
    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=f"Error executing tool {name}: {str(e)}"
            )
        ]

@app.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """
    List available resources
    
    Resources provide data that LLMs can access.
    This example provides a simple configuration resource.
    """
    return [
        types.Resource(
            uri="config://server-info",
            name="Server Information",
            description="Basic information about this MCP server",
            mimeType="application/json"
        )
    ]

@app.read_resource()
async def handle_read_resource(uri: str) -> str:
    """
    Read resource content
    
    This handler returns the actual content of requested resources.
    """
    if uri == "config://server-info":
        server_info = {
            "name": "Basic Example Server",
            "version": "1.0.0",
            "capabilities": ["tools", "resources"],
            "description": "A simple MCP server for learning purposes"
        }
        return json.dumps(server_info, indent=2)
    else:
        raise ValueError(f"Unknown resource: {uri}")

async def main():
    """
    Main server entry point
    
    This function starts the MCP server using stdio transport.
    """
    # Run the server using stdio transport
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="basic-example-server",
                server_version="1.0.0",
                capabilities=app.get_capabilities()
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
```

## Code Annotations

**Line 15:** Server instance creation - this is the main server object
```python
# Create server instance
```

**Line 17:** Tool listing handler - defines what tools are available
```python

```

**Line 45:** Tool execution handler - processes actual tool calls
```python
                "type": "object",
```

**Line 89:** Resource listing handler - defines available data sources
```python
            raise ValueError(f"Unknown tool: {name}")
```

**Line 103:** Resource reading handler - returns actual resource content
```python
    
```

**Line 119:** Main function - sets up stdio transport and runs server
```python
    Read resource content
```

## Dependencies
- mcp

