# MCP Course Code Examples Documentation

This document provides comprehensive documentation for all code examples
used in the MCP Course with Ollama.

## Table of Contents

### Example Collections
- [MCP Fundamentals](#mcp_basics)
- [Ollama Integration](#ollama_integration)

### Individual Examples
- [Basic MCP Server](#basic_mcp_server) (beginner)
- [Basic MCP Client](#basic_mcp_client) (beginner)
- [Ollama MCP Integration](#ollama_integration) (intermediate)

## Example Collections

# MCP Fundamentals - Learning Guide

## Overview
Basic examples covering core MCP concepts and implementation patterns

## Learning Progression

This collection should be studied in the following order:

### 1. Basic MCP Server

**Difficulty:** Beginner

**Description:** A minimal MCP server implementation demonstrating core concepts

**Learning Objectives:**
- Understand basic MCP server structure
- Learn how to define and implement tools
- Understand resource handling
- Learn proper error handling patterns

**Prerequisites:**
- Basic Python knowledge
- Understanding of async/await
- Familiarity with JSON

---

### 2. Basic MCP Client

**Difficulty:** Beginner

**Description:** A simple MCP client that connects to servers and uses tools

**Learning Objectives:**
- Understand MCP client architecture
- Learn how to connect to MCP servers
- Understand tool discovery and execution
- Learn resource access patterns

**Prerequisites:**
- Basic Python knowledge
- Understanding of async/await
- Familiarity with MCP server concepts

---


# Ollama Integration - Learning Guide

## Overview
Examples showing how to integrate MCP with Ollama for enhanced LLM capabilities

## Learning Progression

This collection should be studied in the following order:

### 1. Ollama MCP Integration

**Difficulty:** Intermediate

**Description:** Integration example showing MCP client with Ollama for LLM interactions

**Learning Objectives:**
- Understand Ollama-MCP integration patterns
- Learn how to bridge MCP tools with LLMs
- Understand tool discovery and execution flow
- Learn privacy-preserving AI integration

**Prerequisites:**
- Understanding of MCP clients and servers
- Basic knowledge of HTTP APIs
- Familiarity with Ollama
- JSON processing knowledge

---


## Individual Example Documentation

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


# Ollama MCP Integration

## Description
Integration example showing MCP client with Ollama for LLM interactions

## Learning Objectives
- Understand Ollama-MCP integration patterns
- Learn how to bridge MCP tools with LLMs
- Understand tool discovery and execution flow
- Learn privacy-preserving AI integration

## Prerequisites
- Understanding of MCP clients and servers
- Basic knowledge of HTTP APIs
- Familiarity with Ollama
- JSON processing knowledge

## Key Concepts
- Multi-server MCP management
- Tool discovery and cataloging
- LLM prompt engineering for tools
- Tool call parsing and execution
- Privacy-by-design integration

## Difficulty Level
Intermediate

## Code Example

```python
"""
Ollama MCP Integration Example

This example demonstrates how to integrate MCP tools with Ollama
for enhanced LLM capabilities while maintaining privacy.
"""

import asyncio
import json
import requests
from typing import Dict, List, Any, Optional
from mcp import ClientSession, StdioServerParameters

class OllamaMCPIntegration:
    """
    Integration class for using MCP tools with Ollama
    
    This class bridges MCP servers and Ollama to provide enhanced
    LLM capabilities with local processing.
    """
    
    def __init__(self, 
                 ollama_endpoint: str = "http://localhost:11434",
                 model_name: str = "llama2"):
        """
        Initialize the integration
        
        Args:
            ollama_endpoint: Ollama server endpoint
            model_name: Name of the Ollama model to use
        """
        self.ollama_endpoint = ollama_endpoint
        self.model_name = model_name
        self.mcp_sessions = {}  # Server name -> session
        self.available_tools = {}  # Tool name -> (server_name, tool_info)
    
    async def add_mcp_server(self, server_name: str, command: str, args: List[str] = None):
        """
        Add an MCP server to the integration
        
        Args:
            server_name: Unique name for the server
            command: Command to start the server
            args: Arguments for the server command
        """
        try:
            # Create server parameters
            server_params = StdioServerParameters(
                command=command,
                args=args or []
            )
            
            # Create and initialize session
            session = ClientSession(server_params)
            await session.initialize()
            
            # Store session
            self.mcp_sessions[server_name] = session
            
            # Discover tools from this server
            await self._discover_tools(server_name, session)
            
            print(f"Successfully connected to MCP server: {server_name}")
            
        except Exception as e:
            print(f"Failed to connect to server {server_name}: {e}")
    
    async def _discover_tools(self, server_name: str, session: ClientSession):
        """
        Discover and catalog tools from an MCP server
        
        Args:
            server_name: Name of the server
            session: Active session with the server
        """
        try:
            tools_response = await session.list_tools()
            
            for tool in tools_response.tools:
                tool_key = f"{server_name}:{tool.name}"
                self.available_tools[tool_key] = {
                    'server_name': server_name,
                    'tool_info': tool,
                    'session': session
                }
                
                print(f"Discovered tool: {tool_key} - {tool.description}")
                
        except Exception as e:
            print(f"Error discovering tools from {server_name}: {e}")
    
    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """
        Get tool descriptions formatted for LLM consumption
        
        Returns:
            List of tool descriptions in LLM-friendly format
        """
        tools_description = []
        
        for tool_key, tool_data in self.available_tools.items():
            tool_info = tool_data['tool_info']
            
            description = {
                'name': tool_key,
                'description': tool_info.description,
                'parameters': tool_info.inputSchema if hasattr(tool_info, 'inputSchema') else {}
            }
            
            tools_description.append(description)
        
        return tools_description
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """
        Execute an MCP tool and return the result
        
        Args:
            tool_name: Name of the tool (format: server:tool)
            arguments: Tool arguments
            
        Returns:
            Tool execution result as string
        """
        if tool_name not in self.available_tools:
            return f"Error: Tool '{tool_name}' not found"
        
        try:
            tool_data = self.available_tools[tool_name]
            session = tool_data['session']
            actual_tool_name = tool_name.split(':', 1)[1]  # Remove server prefix
            
            # Execute the tool
            result = await session.call_tool(actual_tool_name, arguments)
            
            # Format result for LLM
            result_text = ""
            for content in result.content:
                if hasattr(content, 'text'):
                    result_text += content.text + "\n"
                else:
                    result_text += str(content) + "\n"
            
            return result_text.strip()
            
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"
    
    async def chat_with_tools(self, user_message: str, system_prompt: str = None) -> str:
        """
        Chat with Ollama using available MCP tools
        
        Args:
            user_message: User's message
            system_prompt: Optional system prompt
            
        Returns:
            LLM response
        """
        # Prepare tools description for the LLM
        tools_info = self.get_tools_for_llm()
        
        # Create enhanced system prompt with tool information
        enhanced_system_prompt = system_prompt or "You are a helpful assistant."
        
        if tools_info:
            tools_description = "\n".join([
                f"- {tool['name']}: {tool['description']}"
                for tool in tools_info
            ])
            
            enhanced_system_prompt += f"""

Available tools:
{tools_description}

To use a tool, respond with a JSON object in this format:
{{"tool_call": {{"name": "tool_name", "arguments": {{"param": "value"}}}}}}

If you don't need to use any tools, respond normally.
"""
        
        # Send request to Ollama
        try:
            response = requests.post(
                f"{self.ollama_endpoint}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": f"System: {enhanced_system_prompt}\n\nUser: {user_message}\n\nAssistant:",
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                llm_response = response.json()['response']
                
                # Check if LLM wants to use a tool
                if "tool_call" in llm_response:
                    try:
                        # Parse tool call
                        tool_call_start = llm_response.find('{"tool_call"')
                        tool_call_end = llm_response.find('}', tool_call_start) + 1
                        tool_call_json = llm_response[tool_call_start:tool_call_end]
                        
                        tool_call = json.loads(tool_call_json)
                        tool_name = tool_call['tool_call']['name']
                        tool_args = tool_call['tool_call']['arguments']
                        
                        # Execute the tool
                        tool_result = await self.execute_tool(tool_name, tool_args)
                        
                        # Get final response with tool result
                        final_prompt = f"""System: {enhanced_system_prompt}

User: {user_message}

Tool used: {tool_name}
Tool result: {tool_result}

Please provide a final response based on the tool result:"""
                        
                        final_response = requests.post(
                            f"{self.ollama_endpoint}/api/generate",
                            json={
                                "model": self.model_name,
                                "prompt": final_prompt,
                                "stream": False
                            }
                        )
                        
                        if final_response.status_code == 200:
                            return final_response.json()['response']
                        else:
                            return f"Tool executed successfully: {tool_result}"
                            
                    except Exception as e:
                        print(f"Error processing tool call: {e}")
                        return llm_response
                
                return llm_response
            else:
                return f"Error from Ollama: {response.status_code}"
                
        except Exception as e:
            return f"Error communicating with Ollama: {str(e)}"
    
    async def cleanup(self):
        """
        Clean up all MCP sessions
        """
        for server_name, session in self.mcp_sessions.items():
            try:
                await session.close()
                print(f"Closed session for {server_name}")
            except Exception as e:
                print(f"Error closing session for {server_name}: {e}")
        
        self.mcp_sessions.clear()
        self.available_tools.clear()

async def demo_ollama_integration():
    """
    Demonstrate Ollama MCP integration
    """
    integration = OllamaMCPIntegration(model_name="llama2")
    
    try:
        # Add MCP servers
        await integration.add_mcp_server(
            "math_server", 
            "python", 
            ["math_server.py"]
        )
        
        await integration.add_mcp_server(
            "file_server",
            "python", 
            ["file_server.py"]
        )
        
        # Chat with tool capabilities
        response = await integration.chat_with_tools(
            "What's 15 multiplied by 23?",
            "You are a helpful math assistant."
        )
        print(f"LLM Response: {response}")
        
        # Another example
        response = await integration.chat_with_tools(
            "Can you read the contents of config.json?",
            "You are a helpful file assistant."
        )
        print(f"LLM Response: {response}")
        
    finally:
        await integration.cleanup()

if __name__ == "__main__":
    asyncio.run(demo_ollama_integration())
```

## Code Annotations

**Line 25:** Integration class initialization with Ollama configuration
```python
        """
```

**Line 45:** MCP server registration and connection management
```python
        """
```

**Line 75:** Tool discovery and cataloging from connected servers
```python
        """
```

**Line 95:** Tool description formatting for LLM consumption
```python
        
```

**Line 115:** Tool execution with proper error handling
```python
        """
```

**Line 145:** Enhanced chat with tool integration capabilities
```python
            
```

**Line 200:** Tool call detection and parsing from LLM response
```python
                        # Parse tool call
```

**Line 230:** Cleanup and resource management
```python
                        
```

## Dependencies
- mcp
- requests


