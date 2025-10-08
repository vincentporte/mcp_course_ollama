"""
Conceptual code examples showing MCP Server/Client interactions.

This module provides practical code examples that demonstrate how MCP
components work together in real-world scenarios.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ToolResult:
    """Represents the result of a tool execution."""
    success: bool
    content: Any
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ConceptualTool(ABC):
    """Abstract base class for conceptual MCP tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def get_schema(self) -> dict[str, Any]:
        """Get the tool's input schema."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    def explain_usage(self) -> str:
        """Explain how to use this tool."""
        return f"Tool '{self.name}': {self.description}"


class FileReadTool(ConceptualTool):
    """Example tool for reading files."""

    def __init__(self):
        super().__init__(
            name="read_file",
            description="Read the contents of a text file"
        )

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            },
            "required": ["path"]
        }

    async def execute(self, **kwargs) -> ToolResult:
        """Simulate reading a file."""
        path = kwargs.get("path")
        if not path:
            return ToolResult(
                success=False,
                content=None,
                error="Path parameter is required"
            )

        # Simulate file reading (in real implementation, would actually read file)
        if path.endswith(".txt"):
            content = f"Contents of {path}:\\nThis is sample file content."
            return ToolResult(
                success=True,
                content=content,
                metadata={"file_type": "text", "size": len(content)}
            )
        else:
            return ToolResult(
                success=False,
                content=None,
                error=f"Unsupported file type for {path}"
            )


class CalculatorTool(ConceptualTool):
    """Example tool for mathematical calculations."""

    def __init__(self):
        super().__init__(
            name="calculate",
            description="Perform mathematical calculations"
        )

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }

    async def execute(self, **kwargs) -> ToolResult:
        """Simulate mathematical calculation."""
        expression = kwargs.get("expression", "")

        # Simple calculation examples (in real implementation, would use safe eval)
        calculations = {
            "2 + 2": 4,
            "10 * 5": 50,
            "100 / 4": 25,
            "2^3": 8
        }

        if expression in calculations:
            result = calculations[expression]
            return ToolResult(
                success=True,
                content=f"{expression} = {result}",
                metadata={"operation": "arithmetic", "result": result}
            )
        else:
            return ToolResult(
                success=False,
                content=None,
                error=f"Unsupported expression: {expression}"
            )


class WeatherTool(ConceptualTool):
    """Example tool for weather information."""

    def __init__(self):
        super().__init__(
            name="get_weather",
            description="Get current weather information for a location"
        )

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City or location name"
                }
            },
            "required": ["location"]
        }

    async def execute(self, **kwargs) -> ToolResult:
        """Simulate weather data retrieval."""
        location = kwargs.get("location", "")

        # Simulate weather data
        weather_data = {
            "new york": {"temp": "22°C", "condition": "Sunny", "humidity": "65%"},
            "london": {"temp": "15°C", "condition": "Cloudy", "humidity": "80%"},
            "tokyo": {"temp": "28°C", "condition": "Partly Cloudy", "humidity": "70%"}
        }

        location_key = location.lower()
        if location_key in weather_data:
            data = weather_data[location_key]
            content = f"Weather in {location}: {data['temp']}, {data['condition']}, Humidity: {data['humidity']}"
            return ToolResult(
                success=True,
                content=content,
                metadata=data
            )
        else:
            return ToolResult(
                success=False,
                content=None,
                error=f"Weather data not available for {location}"
            )


class ConceptualMCPServerExample:
    """Example MCP Server implementation for educational purposes."""

    def __init__(self, name: str):
        self.name = name
        self.tools: dict[str, ConceptualTool] = {}
        self.resources: dict[str, dict[str, Any]] = {}
        self.is_initialized = False
        self.client_info: dict[str, Any] | None = None

    def add_tool(self, tool: ConceptualTool) -> None:
        """Add a tool to the server."""
        self.tools[tool.name] = tool

    def add_resource(self, uri: str, name: str, content: str, mime_type: str = "text/plain") -> None:
        """Add a resource to the server."""
        self.resources[uri] = {
            "name": name,
            "content": content,
            "mimeType": mime_type,
            "uri": uri
        }

    async def handle_initialize(self, client_info: dict[str, Any]) -> dict[str, Any]:
        """Handle client initialization request."""
        self.client_info = client_info
        self.is_initialized = True

        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {
                    "listChanged": True
                },
                "resources": {
                    "subscribe": True,
                    "listChanged": True
                }
            },
            "serverInfo": {
                "name": self.name,
                "version": "1.0.0"
            }
        }

    async def handle_tools_list(self) -> dict[str, Any]:
        """Handle tools list request."""
        tools_list = []
        for tool in self.tools.values():
            tools_list.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.get_schema()
            })

        return {"tools": tools_list}

    async def handle_tool_call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle tool execution request."""
        if name not in self.tools:
            return {
                "error": {
                    "code": -32601,
                    "message": f"Tool '{name}' not found"
                }
            }

        tool = self.tools[name]
        result = await tool.execute(**arguments)

        if result.success:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": str(result.content)
                    }
                ],
                "isError": False
            }
        else:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: {result.error}"
                    }
                ],
                "isError": True
            }

    async def handle_resources_list(self) -> dict[str, Any]:
        """Handle resources list request."""
        resources_list = []
        for uri, resource in self.resources.items():
            resources_list.append({
                "uri": uri,
                "name": resource["name"],
                "mimeType": resource["mimeType"]
            })

        return {"resources": resources_list}

    async def handle_resource_read(self, uri: str) -> dict[str, Any]:
        """Handle resource read request."""
        if uri not in self.resources:
            return {
                "error": {
                    "code": -32602,
                    "message": f"Resource '{uri}' not found"
                }
            }

        resource = self.resources[uri]
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": resource["mimeType"],
                    "text": resource["content"]
                }
            ]
        }

    def demonstrate_server_setup(self) -> list[str]:
        """Demonstrate how to set up an MCP server."""
        setup_steps = [
            "🖥️ MCP Server Setup Example:",
            "",
            f"1. Create server instance: {self.name}",
            f"2. Add tools: {', '.join(self.tools.keys())}",
            f"3. Add resources: {len(self.resources)} resources available",
            "4. Handle client initialization",
            "5. Process tool calls and resource requests",
            "",
            "Server Capabilities:",
            f"  • Tools: {len(self.tools)} available",
            f"  • Resources: {len(self.resources)} available",
            f"  • Status: {'Initialized' if self.is_initialized else 'Not initialized'}"
        ]
        return setup_steps


class ConceptualMCPClientExample:
    """Example MCP Client implementation for educational purposes."""

    def __init__(self, name: str):
        self.name = name
        self.connected_servers: dict[str, ConceptualMCPServerExample] = {}
        self.conversation_context: list[dict[str, Any]] = []

    async def connect_to_server(self, server: ConceptualMCPServerExample) -> bool:
        """Connect to an MCP server."""
        try:
            # Send initialization request
            client_info = {
                "name": self.name,
                "version": "1.0.0"
            }

            await server.handle_initialize(client_info)
            self.connected_servers[server.name] = server

            return True
        except Exception as e:
            print(f"Failed to connect to server {server.name}: {e}")
            return False

    async def discover_capabilities(self, server_name: str) -> dict[str, Any]:
        """Discover capabilities of a connected server."""
        if server_name not in self.connected_servers:
            return {"error": "Server not connected"}

        server = self.connected_servers[server_name]

        tools = await server.handle_tools_list()
        resources = await server.handle_resources_list()

        return {
            "server": server_name,
            "tools": tools.get("tools", []),
            "resources": resources.get("resources", [])
        }

    async def call_tool(self, server_name: str, tool_name: str, **kwargs) -> dict[str, Any]:
        """Call a tool on a specific server."""
        if server_name not in self.connected_servers:
            return {"error": "Server not connected"}

        server = self.connected_servers[server_name]
        result = await server.handle_tool_call(tool_name, kwargs)

        # Add to conversation context
        self.conversation_context.append({
            "type": "tool_call",
            "server": server_name,
            "tool": tool_name,
            "arguments": kwargs,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })

        return result

    async def read_resource(self, server_name: str, uri: str) -> dict[str, Any]:
        """Read a resource from a specific server."""
        if server_name not in self.connected_servers:
            return {"error": "Server not connected"}

        server = self.connected_servers[server_name]
        result = await server.handle_resource_read(uri)

        # Add to conversation context
        self.conversation_context.append({
            "type": "resource_read",
            "server": server_name,
            "uri": uri,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })

        return result

    def get_conversation_summary(self) -> list[str]:
        """Get a summary of the conversation context."""
        summary = [
            "💬 Conversation Context Summary:",
            f"Connected servers: {len(self.connected_servers)}",
            f"Total interactions: {len(self.conversation_context)}",
            ""
        ]

        for i, interaction in enumerate(self.conversation_context[-5:], 1):  # Last 5 interactions
            if interaction["type"] == "tool_call":
                summary.append(f"{i}. Called tool '{interaction['tool']}' on {interaction['server']}")
            elif interaction["type"] == "resource_read":
                summary.append(f"{i}. Read resource '{interaction['uri']}' from {interaction['server']}")

        return summary


class MCPInteractionDemo:
    """Demonstrates complete MCP interactions with realistic scenarios."""

    def __init__(self):
        self.server = None
        self.client = None

    async def setup_demo_environment(self) -> None:
        """Set up a complete demo environment with server and client."""
        # Create and configure server
        self.server = ConceptualMCPServerExample("DemoServer")

        # Add tools
        self.server.add_tool(FileReadTool())
        self.server.add_tool(CalculatorTool())
        self.server.add_tool(WeatherTool())

        # Add resources
        self.server.add_resource(
            "file:///config/settings.json",
            "Application Settings",
            '{"theme": "dark", "language": "en", "notifications": true}',
            "application/json"
        )

        self.server.add_resource(
            "file:///docs/readme.txt",
            "README Documentation",
            "Welcome to the MCP Demo!\\nThis demonstrates MCP capabilities.",
            "text/plain"
        )

        # Create and connect client
        self.client = ConceptualMCPClientExample("DemoClient")
        await self.client.connect_to_server(self.server)

    async def demonstrate_tool_usage_scenario(self) -> list[str]:
        """Demonstrate a realistic tool usage scenario."""
        if not self.server or not self.client:
            await self.setup_demo_environment()

        scenario_steps = [
            "🎯 Scenario: AI Assistant helps with file analysis and calculations",
            ""
        ]

        # Step 1: Discover capabilities
        capabilities = await self.client.discover_capabilities("DemoServer")
        scenario_steps.extend([
            "Step 1: Discover available tools",
            f"  Found {len(capabilities['tools'])} tools:",
            *[f"    • {tool['name']}: {tool['description']}" for tool in capabilities['tools']],
            ""
        ])

        # Step 2: Read a file
        file_result = await self.client.call_tool("DemoServer", "read_file", path="example.txt")
        scenario_steps.extend([
            "Step 2: Read file contents",
            f"  Result: {file_result.get('content', [{}])[0].get('text', 'No content')}",
            ""
        ])

        # Step 3: Perform calculation
        calc_result = await self.client.call_tool("DemoServer", "calculate", expression="2 + 2")
        scenario_steps.extend([
            "Step 3: Perform calculation",
            f"  Result: {calc_result.get('content', [{}])[0].get('text', 'No result')}",
            ""
        ])

        # Step 4: Get weather
        weather_result = await self.client.call_tool("DemoServer", "get_weather", location="New York")
        scenario_steps.extend([
            "Step 4: Get weather information",
            f"  Result: {weather_result.get('content', [{}])[0].get('text', 'No data')}",
            ""
        ])

        # Step 5: Access resource
        resource_result = await self.client.read_resource("DemoServer", "file:///config/settings.json")
        scenario_steps.extend([
            "Step 5: Access configuration resource",
            f"  Content: {resource_result.get('contents', [{}])[0].get('text', 'No content')}",
            ""
        ])

        scenario_steps.extend([
            "🎉 Scenario Complete!",
            "The AI assistant successfully:",
            "  ✅ Connected to MCP server",
            "  ✅ Discovered available capabilities",
            "  ✅ Executed multiple tools",
            "  ✅ Accessed structured resources",
            "  ✅ Maintained conversation context"
        ])

        return scenario_steps

    async def demonstrate_error_handling(self) -> list[str]:
        """Demonstrate error handling in MCP interactions."""
        if not self.server or not self.client:
            await self.setup_demo_environment()

        error_scenarios = [
            "🚨 Error Handling Demonstration:",
            ""
        ]

        # Scenario 1: Invalid tool call
        invalid_tool = await self.client.call_tool("DemoServer", "nonexistent_tool", param="value")
        error_scenarios.extend([
            "Scenario 1: Call non-existent tool",
            f"  Error handled: {invalid_tool.get('content', [{}])[0].get('text', 'Unknown error')}",
            ""
        ])

        # Scenario 2: Missing required parameter
        missing_param = await self.client.call_tool("DemoServer", "read_file")
        error_scenarios.extend([
            "Scenario 2: Missing required parameter",
            f"  Error handled: {missing_param.get('content', [{}])[0].get('text', 'Unknown error')}",
            ""
        ])

        # Scenario 3: Invalid resource URI
        await self.client.read_resource("DemoServer", "file:///nonexistent.txt")
        error_scenarios.extend([
            "Scenario 3: Access non-existent resource",
            "  Error handled: Resource not found",
            ""
        ])

        error_scenarios.extend([
            "Key Error Handling Features:",
            "  ✅ Graceful error responses",
            "  ✅ Descriptive error messages",
            "  ✅ Proper error codes",
            "  ✅ Client-side error detection",
            "  ✅ Conversation context preservation"
        ])

        return error_scenarios

    def get_implementation_insights(self) -> list[str]:
        """Get key insights about MCP implementation."""
        return [
            "💡 MCP Implementation Insights:",
            "",
            "Architecture Benefits:",
            "  • Modular design enables easy extension",
            "  • Standardized protocol ensures interoperability",
            "  • Async operations support concurrent requests",
            "  • Error handling maintains system stability",
            "",
            "Best Practices:",
            "  • Validate all input parameters",
            "  • Provide descriptive error messages",
            "  • Implement proper resource cleanup",
            "  • Maintain conversation context",
            "  • Use appropriate data types in schemas",
            "",
            "Real-world Applications:",
            "  • File system operations",
            "  • Database queries",
            "  • API integrations",
            "  • System monitoring",
            "  • Content generation",
            "  • Development automation"
        ]


# Example usage functions for educational purposes
async def run_complete_demo():
    """Run a complete MCP interaction demo."""
    demo = MCPInteractionDemo()

    print("Setting up demo environment...")
    await demo.setup_demo_environment()

    print("\\nRunning tool usage scenario...")
    scenario_results = await demo.demonstrate_tool_usage_scenario()
    for line in scenario_results:
        print(line)

    print("\\nDemonstrating error handling...")
    error_results = await demo.demonstrate_error_handling()
    for line in error_results:
        print(line)

    print("\\nImplementation insights:")
    insights = demo.get_implementation_insights()
    for line in insights:
        print(line)


def create_learning_examples() -> dict[str, Any]:
    """Create a collection of learning examples for MCP concepts."""
    return {
        "basic_server": {
            "description": "Simple MCP server with file operations",
            "code": """
# Basic MCP Server Example
server = ConceptualMCPServerExample("FileServer")
server.add_tool(FileReadTool())
server.add_resource("file:///data.txt", "Sample Data", "Hello, MCP!")
            """,
            "explanation": "Creates a server with file reading capability and a sample resource"
        },
        "client_connection": {
            "description": "Client connecting to server and discovering tools",
            "code": """
# Client Connection Example
client = ConceptualMCPClientExample("MyClient")
await client.connect_to_server(server)
capabilities = await client.discover_capabilities("FileServer")
            """,
            "explanation": "Establishes connection and discovers available server capabilities"
        },
        "tool_execution": {
            "description": "Executing tools through MCP client",
            "code": """
# Tool Execution Example
result = await client.call_tool("FileServer", "read_file", path="example.txt")
print(result['content'][0]['text'])
            """,
            "explanation": "Calls a tool on the server and processes the result"
        }
    }
