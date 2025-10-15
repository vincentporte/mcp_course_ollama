"""Ollama-MCP bridge for seamless integration between MCP Clients and Ollama LLMs."""

from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from typing import TYPE_CHECKING, Any

from mcp.types import CallToolResult, TextContent

from mcp_course.client.basic import BasicMCPClient
from mcp_course.ollama_client.client import OllamaClient
from mcp_course.ollama_client.config import OllamaConfig


if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class MCPToolDefinition:
    """Definition of an MCP tool for LLM consumption."""
    name: str
    description: str
    parameters: dict[str, Any]
    server_name: str

    def to_ollama_function(self) -> dict[str, Any]:
        """Convert MCP tool to Ollama function definition."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


@dataclass
class ConversationContext:
    """Context for managing conversation state."""
    messages: list[dict[str, Any]] = field(default_factory=list)
    available_tools: list[MCPToolDefinition] = field(default_factory=list)
    tool_results: dict[str, Any] = field(default_factory=dict)
    conversation_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)


class OllamaMCPBridge:
    """
    Bridge between Ollama LLMs and MCP Servers.

    This class provides seamless integration allowing LLMs hosted by Ollama
    to discover and use tools from MCP Servers. It handles:
    - Tool discovery and registration
    - Function calling integration
    - Response formatting and parsing
    - Error handling and recovery
    """

    def __init__(
        self,
        mcp_client: BasicMCPClient,
        ollama_client: OllamaClient = None,
        ollama_config: OllamaConfig = None
    ):
        """Initialize the Ollama-MCP bridge."""
        self.mcp_client = mcp_client
        self.ollama_client = ollama_client or OllamaClient(ollama_config or OllamaConfig())
        self.logger = logging.getLogger("OllamaMCPBridge")
        self.available_tools: dict[str, MCPToolDefinition] = {}
        self.tool_call_handlers: dict[str, Callable] = {}

    async def discover_tools(self, server_names: list[str] | None = None) -> int:
        """
        Discover available tools from MCP Servers.

        Args:
            server_names: List of server names to discover from, or None for all

        Returns:
            Number of tools discovered
        """
        servers_to_check = server_names or self.mcp_client.get_connected_servers()
        tools_discovered = 0

        self.logger.info(f"Discovering tools from {len(servers_to_check)} servers")

        for server_name in servers_to_check:
            try:
                tools = await self.mcp_client.list_server_tools(server_name)
                if tools:
                    for tool in tools:
                        tool_def = MCPToolDefinition(
                            name=tool.name,
                            description=tool.description,
                            parameters=tool.inputSchema,
                            server_name=server_name
                        )

                        # Use server_name.tool_name as unique key
                        tool_key = f"{server_name}.{tool.name}"
                        self.available_tools[tool_key] = tool_def
                        tools_discovered += 1

                        self.logger.debug(f"Discovered tool: {tool_key}")

            except Exception as e:
                self.logger.error(f"Error discovering tools from server {server_name}: {e}")

        self.logger.info(f"Discovered {tools_discovered} tools total")
        return tools_discovered

    def get_ollama_functions(self) -> list[dict[str, Any]]:
        """
        Get tool definitions formatted for Ollama function calling.

        Returns:
            List of function definitions for Ollama
        """
        return [tool.to_ollama_function() for tool in self.available_tools.values()]

    async def call_mcp_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any]
    ) -> CallToolResult | None:
        """
        Call an MCP tool by name.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool execution result or None if error
        """
        # Find the tool definition
        tool_def = None
        for _, tool in self.available_tools.items():
            if tool.name == tool_name:
                tool_def = tool
                break

        if not tool_def:
            self.logger.error(f"Tool not found: {tool_name}")
            return None

        try:
            result = await self.mcp_client.call_tool(
                tool_def.server_name,
                tool_name,
                arguments
            )

            if result:
                self.logger.info(f"Successfully called MCP tool: {tool_name}")

            return result

        except Exception as e:
            self.logger.error(f"Error calling MCP tool {tool_name}: {e}")
            return None

    async def chat_with_tools(
        self,
        message: str,
        model: str | None = None,
        context: ConversationContext = None,
        max_tool_calls: int = 5,
        system_prompt: str | None = None,
        stream: bool = False
    ) -> dict[str, Any]:
        """
        Chat with Ollama LLM with MCP tool integration.

        Args:
            message: User message
            model: Ollama model name
            context: Conversation context
            max_tool_calls: Maximum number of tool calls per turn
            system_prompt: Optional system prompt for the conversation
            stream: Whether to stream the response

        Returns:
            Dictionary containing response and tool call information
        """
        if context is None:
            context = ConversationContext()

        # Update available tools in context
        await self.discover_tools()
        context.available_tools = list(self.available_tools.values())

        # Add user message to context
        context.messages.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })

        model_name = model or self.ollama_client.config.model_name
        tool_calls_made = 0

        try:
            # Prepare messages for Ollama
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add conversation messages
            for msg in context.messages:
                messages.append({"role": msg["role"], "content": msg["content"]})

            # Prepare function definitions for Ollama (if model supports function calling)
            functions = None
            if self.available_tools and await self._model_supports_functions(model_name):
                functions = self.get_ollama_functions()

            # Make initial request to Ollama
            response = await self.ollama_client.chat(
                model=model_name,
                messages=messages,
                functions=functions,
                stream=stream
            )

            if stream:
                # Handle streaming response
                return await self._handle_streaming_response(response, context, max_tool_calls)
            
            assistant_message = response.get("message", {})

            # Handle function calls if present
            while (assistant_message.get("tool_calls") and
                   tool_calls_made < max_tool_calls):

                tool_calls = assistant_message.get("tool_calls", [])
                tool_results = []

                for tool_call in tool_calls:
                    function_name = tool_call.get("function", {}).get("name")
                    function_args = tool_call.get("function", {}).get("arguments", {})

                    if isinstance(function_args, str):
                        try:
                            function_args = json.loads(function_args)
                        except json.JSONDecodeError:
                            self.logger.error(f"Invalid function arguments: {function_args}")
                            continue

                    # Call the MCP tool
                    tool_result = await self.call_mcp_tool(function_name, function_args)

                    if tool_result:
                        # Extract text content from tool result
                        result_text = ""
                        for content in tool_result.content:
                            if isinstance(content, TextContent):
                                result_text += content.text

                        tool_results.append({
                            "tool_call_id": tool_call.get("id", ""),
                            "function_name": function_name,
                            "result": result_text
                        })

                        # Store in context
                        context.tool_results[function_name] = result_text

                # Add tool results to conversation
                if tool_results:
                    context.messages.append({
                        "role": "tool",
                        "content": json.dumps(tool_results),
                        "timestamp": datetime.now().isoformat()
                    })

                    # Continue conversation with tool results
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    
                    for msg in context.messages:
                        messages.append({"role": msg["role"], "content": msg["content"]})

                    response = await self.ollama_client.chat(
                        model=model_name,
                        messages=messages,
                        functions=functions,
                        stream=False
                    )

                    assistant_message = response.get("message", {})
                    tool_calls_made += 1
                else:
                    break

            # Add final assistant response to context
            final_content = assistant_message.get("content", "")
            context.messages.append({
                "role": "assistant",
                "content": final_content,
                "timestamp": datetime.now().isoformat()
            })

            return {
                "response": final_content,
                "tool_calls_made": tool_calls_made,
                "tools_used": list(context.tool_results.keys()),
                "context": context,
                "success": True,
                "model_used": model_name
            }

        except Exception as e:
            self.logger.error(f"Error in chat with tools: {e}")
            return {
                "response": f"Error: {e}",
                "tool_calls_made": tool_calls_made,
                "tools_used": [],
                "context": context,
                "success": False,
                "error": str(e),
                "model_used": model_name
            }

    def create_tool_aware_prompt(
        self,
        user_message: str,
        include_tool_list: bool = True
    ) -> str:
        """
        Create a prompt that makes the LLM aware of available MCP tools.

        Args:
            user_message: Original user message
            include_tool_list: Whether to include list of available tools

        Returns:
            Enhanced prompt with tool awareness
        """
        if not self.available_tools or not include_tool_list:
            return user_message

        tool_descriptions = []
        for tool in self.available_tools.values():
            tool_descriptions.append(f"- {tool.name}: {tool.description}")

        enhanced_prompt = f"""You have access to the following MCP tools:

{chr(10).join(tool_descriptions)}

User request: {user_message}

You can use these tools to help answer the user's request. Call the appropriate tools when needed."""

        return enhanced_prompt

    async def _model_supports_functions(self, model_name: str) -> bool:
        """
        Check if the model supports function calling.

        Args:
            model_name: Name of the model to check

        Returns:
            True if model supports function calling
        """
        # For now, assume most modern models support function calling
        # This could be enhanced to check model capabilities
        function_capable_models = [
            "llama3.1", "llama3.2", "mistral", "codellama", "qwen", "gemma"
        ]
        
        return any(model_type in model_name.lower() for model_type in function_capable_models)

    async def _handle_streaming_response(
        self,
        response_stream,
        context: ConversationContext,
        max_tool_calls: int
    ) -> dict[str, Any]:
        """
        Handle streaming response from Ollama.

        Args:
            response_stream: Streaming response from Ollama
            context: Conversation context
            max_tool_calls: Maximum tool calls allowed

        Returns:
            Response dictionary with streaming results
        """
        full_response = ""
        tool_calls_made = 0

        try:
            async for chunk in response_stream:
                if chunk.get("message", {}).get("content"):
                    content = chunk["message"]["content"]
                    full_response += content
                    
                # Handle tool calls in streaming mode if present
                if chunk.get("message", {}).get("tool_calls"):
                    # For streaming, we'd need to handle tool calls as they come
                    # This is a simplified implementation
                    pass

            # Add final response to context
            context.messages.append({
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.now().isoformat()
            })

            return {
                "response": full_response,
                "tool_calls_made": tool_calls_made,
                "tools_used": list(context.tool_results.keys()),
                "context": context,
                "success": True,
                "streaming": True
            }

        except Exception as e:
            self.logger.error(f"Error in streaming response: {e}")
            return {
                "response": f"Streaming error: {e}",
                "tool_calls_made": tool_calls_made,
                "tools_used": [],
                "context": context,
                "success": False,
                "error": str(e),
                "streaming": True
            }

    async def batch_tool_calls(
        self,
        tool_requests: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Execute multiple tool calls in batch.

        Args:
            tool_requests: List of tool request dictionaries with 'name' and 'arguments'

        Returns:
            List of tool results
        """
        results = []
        
        for request in tool_requests:
            tool_name = request.get("name")
            arguments = request.get("arguments", {})
            
            if not tool_name:
                results.append({
                    "error": "Missing tool name",
                    "success": False
                })
                continue
                
            result = await self.call_mcp_tool(tool_name, arguments)
            
            if result:
                # Extract text content
                result_text = ""
                for content in result.content:
                    if isinstance(content, TextContent):
                        result_text += content.text
                        
                results.append({
                    "tool_name": tool_name,
                    "result": result_text,
                    "success": True
                })
            else:
                results.append({
                    "tool_name": tool_name,
                    "error": "Tool execution failed",
                    "success": False
                })
                
        return results

    def get_tool_usage_stats(self) -> dict[str, Any]:
        """
        Get statistics about tool usage.

        Returns:
            Dictionary with tool usage statistics
        """
        return {
            "total_tools_available": len(self.available_tools),
            "tools_by_server": self._get_tools_by_server(),
            "tool_categories": self._categorize_tools()
        }

    def _get_tools_by_server(self) -> dict[str, int]:
        """Get count of tools by server."""
        server_counts = {}
        for tool in self.available_tools.values():
            server_counts[tool.server_name] = server_counts.get(tool.server_name, 0) + 1
        return server_counts

    def _categorize_tools(self) -> dict[str, list[str]]:
        """Categorize tools by type based on their names and descriptions."""
        categories = {
            "data": [],
            "communication": [],
            "file_system": [],
            "calculation": [],
            "web": [],
            "other": []
        }
        
        for tool in self.available_tools.values():
            name_lower = tool.name.lower()
            desc_lower = tool.description.lower()
            
            if any(keyword in name_lower or keyword in desc_lower 
                   for keyword in ["data", "database", "query", "analyze"]):
                categories["data"].append(tool.name)
            elif any(keyword in name_lower or keyword in desc_lower 
                     for keyword in ["email", "send", "notify", "message"]):
                categories["communication"].append(tool.name)
            elif any(keyword in name_lower or keyword in desc_lower 
                     for keyword in ["file", "directory", "path", "search"]):
                categories["file_system"].append(tool.name)
            elif any(keyword in name_lower or keyword in desc_lower 
                     for keyword in ["calc", "math", "compute", "calculate"]):
                categories["calculation"].append(tool.name)
            elif any(keyword in name_lower or keyword in desc_lower 
                     for keyword in ["web", "http", "url", "api"]):
                categories["web"].append(tool.name)
            else:
                categories["other"].append(tool.name)
                
        return categories
