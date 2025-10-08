"""
Interactive MCP concept explanations and demonstrations.

This module provides Python classes that demonstrate MCP architecture concepts
with visual diagrams and conceptual code examples.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Any


class MCPComponentType(Enum):
    """Types of MCP components."""
    SERVER = "server"
    CLIENT = "client"
    TOOL = "tool"
    RESOURCE = "resource"
    PROMPT = "prompt"


class MessageType(Enum):
    """MCP protocol message types."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"


@dataclass
class MCPMessage:
    """Represents an MCP protocol message for educational purposes."""

    message_type: MessageType
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    id: str | None = None
    result: Any | None = None
    error: dict[str, Any] | None = None

    def to_json(self) -> str:
        """Convert message to JSON representation."""
        message_dict = {
            "jsonrpc": "2.0",
            "method": self.method
        }

        if self.id is not None:
            message_dict["id"] = self.id

        if self.params:
            message_dict["params"] = self.params

        if self.result is not None:
            message_dict["result"] = self.result

        if self.error:
            message_dict["error"] = self.error

        return json.dumps(message_dict, indent=2)

    def explain(self) -> str:
        """Provide educational explanation of the message."""
        explanations = {
            "initialize": "Establishes connection between client and server, exchanging capabilities",
            "tools/list": "Client requests list of available tools from server",
            "tools/call": "Client invokes a specific tool with parameters",
            "resources/list": "Client requests list of available resources from server",
            "resources/read": "Client requests to read content from a specific resource",
            "prompts/list": "Client requests list of available prompts from server",
            "prompts/get": "Client requests a specific prompt template"
        }

        base_explanation = explanations.get(self.method, f"MCP method: {self.method}")

        if self.message_type == MessageType.REQUEST:
            return f"📤 REQUEST: {base_explanation}"
        elif self.message_type == MessageType.RESPONSE:
            return f"📥 RESPONSE: Response to {base_explanation}"
        else:
            return f"🔔 NOTIFICATION: {base_explanation}"


@dataclass
class MCPTool:
    """Represents an MCP tool for educational demonstrations."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def create_call_example(self, **kwargs) -> MCPMessage:
        """Create an example tool call message."""
        return MCPMessage(
            message_type=MessageType.REQUEST,
            method="tools/call",
            id="call-1",
            params={
                "name": self.name,
                "arguments": kwargs
            }
        )

    def create_response_example(self, content: Any) -> MCPMessage:
        """Create an example tool response message."""
        return MCPMessage(
            message_type=MessageType.RESPONSE,
            method="tools/call",
            id="call-1",
            result={
                "content": [
                    {
                        "type": "text",
                        "text": str(content)
                    }
                ]
            }
        )

    def explain_purpose(self) -> str:
        """Explain the tool's purpose and usage."""
        return f"🔧 Tool '{self.name}': {self.description}"


@dataclass
class MCPResource:
    """Represents an MCP resource for educational demonstrations."""

    uri: str
    name: str
    description: str
    mime_type: str = "text/plain"

    def create_read_example(self) -> MCPMessage:
        """Create an example resource read message."""
        return MCPMessage(
            message_type=MessageType.REQUEST,
            method="resources/read",
            id="read-1",
            params={"uri": self.uri}
        )

    def create_response_example(self, content: str) -> MCPMessage:
        """Create an example resource response message."""
        return MCPMessage(
            message_type=MessageType.RESPONSE,
            method="resources/read",
            id="read-1",
            result={
                "contents": [
                    {
                        "uri": self.uri,
                        "mimeType": self.mime_type,
                        "text": content
                    }
                ]
            }
        )

    def explain_purpose(self) -> str:
        """Explain the resource's purpose and usage."""
        return f"📄 Resource '{self.name}': {self.description}"


class MCPComponent(ABC):
    """Abstract base class for MCP components."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.component_type = self._get_component_type()

    @abstractmethod
    def _get_component_type(self) -> MCPComponentType:
        """Get the component type."""
        pass

    @abstractmethod
    def get_capabilities(self) -> dict[str, Any]:
        """Get component capabilities."""
        pass

    def explain_role(self) -> str:
        """Explain the component's role in MCP architecture."""
        role_explanations = {
            MCPComponentType.SERVER: "Provides tools, resources, and prompts to clients",
            MCPComponentType.CLIENT: "Connects to servers and uses their capabilities",
            MCPComponentType.TOOL: "Executable function that extends LLM capabilities",
            MCPComponentType.RESOURCE: "Structured data accessible to LLMs",
            MCPComponentType.PROMPT: "Template for generating LLM prompts"
        }
        return f"{self.component_type.value.upper()}: {role_explanations[self.component_type]}"


class ConceptualMCPServer(MCPComponent):
    """Educational representation of an MCP Server."""

    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.tools: list[MCPTool] = []
        self.resources: list[MCPResource] = []
        self.is_connected = False

    def _get_component_type(self) -> MCPComponentType:
        return MCPComponentType.SERVER

    def add_tool(self, tool: MCPTool) -> None:
        """Add a tool to the server."""
        self.tools.append(tool)

    def add_resource(self, resource: MCPResource) -> None:
        """Add a resource to the server."""
        self.resources.append(resource)

    def get_capabilities(self) -> dict[str, Any]:
        """Get server capabilities."""
        return {
            "tools": {tool.name: tool.description for tool in self.tools},
            "resources": {resource.uri: resource.description for resource in self.resources},
            "experimental": {},
            "roots": {
                "listChanged": True
            }
        }

    def create_initialization_response(self) -> MCPMessage:
        """Create initialization response message."""
        return MCPMessage(
            message_type=MessageType.RESPONSE,
            method="initialize",
            id="init-1",
            result={
                "protocolVersion": "2024-11-05",
                "capabilities": self.get_capabilities(),
                "serverInfo": {
                    "name": self.name,
                    "version": "1.0.0"
                }
            }
        )

    def create_tools_list_response(self) -> MCPMessage:
        """Create tools list response message."""
        return MCPMessage(
            message_type=MessageType.RESPONSE,
            method="tools/list",
            id="tools-1",
            result={
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.input_schema
                    }
                    for tool in self.tools
                ]
            }
        )

    def demonstrate_interaction_flow(self) -> list[str]:
        """Demonstrate typical server interaction flow."""
        flow = [
            "🔄 MCP Server Interaction Flow:",
            "1. Client sends initialization request",
            "2. Server responds with capabilities and info",
            "3. Client requests available tools/resources",
            "4. Server provides lists of available capabilities",
            "5. Client calls tools or reads resources as needed",
            "6. Server executes requests and returns results"
        ]
        return flow


class ConceptualMCPClient(MCPComponent):
    """Educational representation of an MCP Client."""

    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.connected_servers: list[ConceptualMCPServer] = []
        self.conversation_history: list[dict[str, Any]] = []

    def _get_component_type(self) -> MCPComponentType:
        return MCPComponentType.CLIENT

    def connect_to_server(self, server: ConceptualMCPServer) -> None:
        """Connect to an MCP server."""
        if server not in self.connected_servers:
            self.connected_servers.append(server)
            server.is_connected = True

    def get_capabilities(self) -> dict[str, Any]:
        """Get client capabilities."""
        return {
            "experimental": {},
            "sampling": {},
            "roots": {
                "listChanged": True
            }
        }

    def create_initialization_request(self) -> MCPMessage:
        """Create initialization request message."""
        return MCPMessage(
            message_type=MessageType.REQUEST,
            method="initialize",
            id="init-1",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": self.get_capabilities(),
                "clientInfo": {
                    "name": self.name,
                    "version": "1.0.0"
                }
            }
        )

    def create_tools_list_request(self) -> MCPMessage:
        """Create tools list request message."""
        return MCPMessage(
            message_type=MessageType.REQUEST,
            method="tools/list",
            id="tools-1"
        )

    def get_available_tools(self) -> list[MCPTool]:
        """Get all available tools from connected servers."""
        tools = []
        for server in self.connected_servers:
            tools.extend(server.tools)
        return tools

    def get_available_resources(self) -> list[MCPResource]:
        """Get all available resources from connected servers."""
        resources = []
        for server in self.connected_servers:
            resources.extend(server.resources)
        return resources

    def demonstrate_usage_pattern(self) -> list[str]:
        """Demonstrate typical client usage pattern."""
        pattern = [
            "🔄 MCP Client Usage Pattern:",
            "1. Initialize connection with server(s)",
            "2. Discover available tools and resources",
            "3. Integrate capabilities with LLM conversations",
            "4. Call tools based on LLM requests",
            "5. Provide tool results back to LLM",
            "6. Continue conversation with enhanced context"
        ]
        return pattern


class MCPArchitectureDemonstrator:
    """Demonstrates MCP architecture concepts with interactive examples."""

    def __init__(self):
        self.servers: list[ConceptualMCPServer] = []
        self.clients: list[ConceptualMCPClient] = []

    def create_example_ecosystem(self) -> None:
        """Create a complete example MCP ecosystem."""
        # Create example server
        file_server = ConceptualMCPServer(
            name="FileSystemServer",
            description="Provides file system access tools and resources"
        )

        # Add example tools
        read_file_tool = MCPTool(
            name="read_file",
            description="Read contents of a file",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"}
                },
                "required": ["path"]
            }
        )

        write_file_tool = MCPTool(
            name="write_file",
            description="Write content to a file",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["path", "content"]
            }
        )

        file_server.add_tool(read_file_tool)
        file_server.add_tool(write_file_tool)

        # Add example resources
        config_resource = MCPResource(
            uri="file:///config/app.json",
            name="Application Configuration",
            description="Main application configuration file",
            mime_type="application/json"
        )

        file_server.add_resource(config_resource)

        # Create example client
        ai_assistant = ConceptualMCPClient(
            name="AIAssistant",
            description="AI assistant that uses MCP servers for enhanced capabilities"
        )

        # Connect client to server
        ai_assistant.connect_to_server(file_server)

        self.servers.append(file_server)
        self.clients.append(ai_assistant)

    def demonstrate_complete_interaction(self) -> list[str]:
        """Demonstrate a complete MCP interaction sequence."""
        if not self.servers or not self.clients:
            self.create_example_ecosystem()

        server = self.servers[0]
        client = self.clients[0]

        interaction_steps = [
            "🎯 Complete MCP Interaction Demonstration",
            "",
            "📋 Scenario: AI Assistant needs to read a configuration file",
            "",
            "Step 1: Client Initialization",
            f"  {client.create_initialization_request().explain()}",
            "",
            "Step 2: Server Response",
            f"  {server.create_initialization_response().explain()}",
            "",
            "Step 3: Client Discovers Tools",
            f"  {client.create_tools_list_request().explain()}",
            "",
            "Step 4: Server Lists Available Tools",
            f"  {server.create_tools_list_response().explain()}",
            "",
            "Step 5: Client Calls Tool",
            f"  {server.tools[0].create_call_example(path='/config/app.json').explain()}",
            "",
            "Step 6: Server Executes and Responds",
            f"  {server.tools[0].create_response_example('Configuration loaded successfully').explain()}",
            "",
            "🎉 Result: AI Assistant now has access to configuration data!"
        ]

        return interaction_steps

    def explain_component_relationships(self) -> list[str]:
        """Explain how MCP components relate to each other."""
        relationships = [
            "🏗️ MCP Component Relationships:",
            "",
            "┌─────────────┐    ┌─────────────┐",
            "│   Client    │◄──►│   Server    │",
            "│             │    │             │",
            "│ - Connects  │    │ - Provides  │",
            "│ - Requests  │    │ - Executes  │",
            "│ - Integrates│    │ - Responds  │",
            "└─────────────┘    └─────────────┘",
            "       │                   │",
            "       │                   │",
            "       ▼                   ▼",
            "┌─────────────┐    ┌─────────────┐",
            "│     LLM     │    │Tools/Resources│",
            "│             │    │             │",
            "│ - Processes │    │ - Functions │",
            "│ - Generates │    │ - Data      │",
            "│ - Responds  │    │ - Templates │",
            "└─────────────┘    └─────────────┘",
            "",
            "Key Relationships:",
            "• Client ↔ Server: Bidirectional communication via JSON-RPC",
            "• Server → Tools/Resources: Server exposes capabilities",
            "• Client → LLM: Client integrates MCP capabilities with LLM",
            "• LLM → Client: LLM requests tool usage through client"
        ]

        return relationships

    def get_learning_summary(self) -> dict[str, list[str]]:
        """Get a comprehensive learning summary of MCP concepts."""
        return {
            "core_concepts": [
                "MCP enables LLMs to access external tools and data",
                "Servers provide capabilities, clients consume them",
                "Communication uses JSON-RPC protocol",
                "Tools are executable functions",
                "Resources are structured data sources",
                "Prompts are reusable templates"
            ],
            "architecture_benefits": [
                "Modular design allows flexible integrations",
                "Standardized protocol ensures interoperability",
                "Security through controlled access patterns",
                "Scalability via distributed server architecture",
                "Extensibility through custom tool development"
            ],
            "practical_applications": [
                "File system access for AI assistants",
                "Database queries and data analysis",
                "API integrations and web services",
                "Development tool automation",
                "Content management and generation",
                "System monitoring and administration"
            ]
        }
