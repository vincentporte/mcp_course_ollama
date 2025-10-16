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

