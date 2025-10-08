"""
MCP protocol demonstration utilities.

This module provides mock MCP Server and Client implementations for educational
purposes, along with protocol message examples and interactive demonstrations.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
from typing import Any


class MCPProtocolVersion(Enum):
    """Supported MCP protocol versions."""
    V2024_11_05 = "2024-11-05"


class JSONRPCErrorCode(Enum):
    """Standard JSON-RPC error codes."""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


@dataclass
class JSONRPCError:
    """Represents a JSON-RPC error."""
    code: int
    message: str
    data: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary."""
        error_dict = {
            "code": self.code,
            "message": self.message
        }
        if self.data is not None:
            error_dict["data"] = self.data
        return error_dict


@dataclass
class JSONRPCMessage:
    """Represents a JSON-RPC 2.0 message."""
    jsonrpc: str = "2.0"
    method: str | None = None
    params: dict[str, Any] | None = None
    id: str | int | None = None
    result: Any | None = None
    error: JSONRPCError | None = None

    def is_request(self) -> bool:
        """Check if message is a request."""
        return self.method is not None and self.id is not None

    def is_notification(self) -> bool:
        """Check if message is a notification."""
        return self.method is not None and self.id is None

    def is_response(self) -> bool:
        """Check if message is a response."""
        return self.method is None and self.id is not None

    def to_json(self) -> str:
        """Convert message to JSON string."""
        message_dict = {"jsonrpc": self.jsonrpc}

        if self.method is not None:
            message_dict["method"] = self.method

        if self.params is not None:
            message_dict["params"] = self.params

        if self.id is not None:
            message_dict["id"] = self.id

        if self.result is not None:
            message_dict["result"] = self.result

        if self.error is not None:
            message_dict["error"] = self.error.to_dict()

        return json.dumps(message_dict, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'JSONRPCMessage':
        """Create message from JSON string."""
        data = json.loads(json_str)

        error = None
        if "error" in data:
            error_data = data["error"]
            error = JSONRPCError(
                code=error_data["code"],
                message=error_data["message"],
                data=error_data.get("data")
            )

        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            method=data.get("method"),
            params=data.get("params"),
            id=data.get("id"),
            result=data.get("result"),
            error=error
        )

    def explain_structure(self) -> list[str]:
        """Explain the message structure for educational purposes."""
        explanations = [
            "📋 JSON-RPC Message Structure Analysis:",
            f"  • Protocol Version: {self.jsonrpc}",
        ]

        if self.is_request():
            explanations.extend([
                "  • Type: Request",
                f"  • Method: {self.method}",
                f"  • ID: {self.id} (for matching response)",
                f"  • Parameters: {len(self.params or {})} provided"
            ])
        elif self.is_notification():
            explanations.extend([
                "  • Type: Notification",
                f"  • Method: {self.method}",
                "  • No ID (no response expected)"
            ])
        elif self.is_response():
            explanations.extend([
                "  • Type: Response",
                f"  • ID: {self.id} (matches request)",
                f"  • Success: {'Yes' if self.result is not None else 'No'}",
                f"  • Error: {'Yes' if self.error is not None else 'No'}"
            ])

        return explanations


class MCPCapabilities:
    """Represents MCP capabilities for servers and clients."""

    def __init__(self):
        self.experimental: dict[str, Any] = {}
        self.sampling: dict[str, Any] = {}
        self.tools: dict[str, Any] = {}
        self.resources: dict[str, Any] = {}
        self.prompts: dict[str, Any] = {}
        self.roots: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any]:
        """Convert capabilities to dictionary."""
        caps = {}

        if self.experimental:
            caps["experimental"] = self.experimental
        if self.sampling:
            caps["sampling"] = self.sampling
        if self.tools:
            caps["tools"] = self.tools
        if self.resources:
            caps["resources"] = self.resources
        if self.prompts:
            caps["prompts"] = self.prompts
        if self.roots:
            caps["roots"] = self.roots

        return caps


class MockMCPServer:
    """Mock MCP Server for educational demonstrations."""

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.protocol_version = MCPProtocolVersion.V2024_11_05.value
        self.capabilities = MCPCapabilities()
        self.tools: dict[str, dict[str, Any]] = {}
        self.resources: dict[str, dict[str, Any]] = {}
        self.prompts: dict[str, dict[str, Any]] = {}
        self.is_initialized = False
        self.client_info: dict[str, Any] | None = None
        self.message_log: list[JSONRPCMessage] = []

        # Set up default capabilities
        self.capabilities.tools = {"listChanged": True}
        self.capabilities.resources = {"subscribe": True, "listChanged": True}
        self.capabilities.prompts = {"listChanged": True}

    def add_tool(self, name: str, description: str, input_schema: dict[str, Any]) -> None:
        """Add a tool to the server."""
        self.tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": input_schema
        }

    def add_resource(self, uri: str, name: str, description: str, mime_type: str = "text/plain") -> None:
        """Add a resource to the server."""
        self.resources[uri] = {
            "uri": uri,
            "name": name,
            "description": description,
            "mimeType": mime_type
        }

    def add_prompt(self, name: str, description: str, arguments: list[dict[str, Any]] | None = None) -> None:
        """Add a prompt to the server."""
        self.prompts[name] = {
            "name": name,
            "description": description,
            "arguments": arguments or []
        }

    async def handle_message(self, message: JSONRPCMessage) -> JSONRPCMessage | None:
        """Handle incoming message and return response if needed."""
        self.message_log.append(message)

        if message.is_request():
            return await self._handle_request(message)
        elif message.is_notification():
            await self._handle_notification(message)
            return None
        else:
            # This is a response, which servers don't typically handle
            return None

    async def _handle_request(self, message: JSONRPCMessage) -> JSONRPCMessage:
        """Handle request message."""
        method = message.method
        params = message.params or {}

        try:
            if method == "initialize":
                result = await self._handle_initialize(params)
            elif method == "tools/list":
                result = await self._handle_tools_list(params)
            elif method == "tools/call":
                result = await self._handle_tool_call(params)
            elif method == "resources/list":
                result = await self._handle_resources_list(params)
            elif method == "resources/read":
                result = await self._handle_resource_read(params)
            elif method == "prompts/list":
                result = await self._handle_prompts_list(params)
            elif method == "prompts/get":
                result = await self._handle_prompt_get(params)
            else:
                raise Exception(f"Method not found: {method}")

            response = JSONRPCMessage(
                id=message.id,
                result=result
            )

        except Exception as e:
            response = JSONRPCMessage(
                id=message.id,
                error=JSONRPCError(
                    code=JSONRPCErrorCode.INTERNAL_ERROR.value,
                    message=str(e)
                )
            )

        self.message_log.append(response)
        return response

    async def _handle_notification(self, message: JSONRPCMessage) -> None:
        """Handle notification message."""
        # Notifications don't require responses
        pass

    async def _handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle initialization request."""
        self.client_info = params
        self.is_initialized = True

        return {
            "protocolVersion": self.protocol_version,
            "capabilities": self.capabilities.to_dict(),
            "serverInfo": {
                "name": self.name,
                "version": self.version
            }
        }

    async def _handle_tools_list(self) -> dict[str, Any]:
        """Handle tools list request."""
        return {
            "tools": list(self.tools.values())
        }

    async def _handle_tool_call(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tool call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name not in self.tools:
            raise Exception(f"Tool '{tool_name}' not found")

        # Simulate tool execution
        result = await self._simulate_tool_execution(tool_name, arguments)

        return {
            "content": [
                {
                    "type": "text",
                    "text": result
                }
            ]
        }

    async def _handle_resources_list(self) -> dict[str, Any]:
        """Handle resources list request."""
        return {
            "resources": [
                {
                    "uri": resource["uri"],
                    "name": resource["name"],
                    "description": resource.get("description", ""),
                    "mimeType": resource.get("mimeType", "text/plain")
                }
                for resource in self.resources.values()
            ]
        }

    async def _handle_resource_read(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle resource read request."""
        uri = params.get("uri")

        if uri not in self.resources:
            raise Exception(f"Resource '{uri}' not found")

        resource = self.resources[uri]

        # Simulate resource content
        content = await self._simulate_resource_content(uri)

        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": resource["mimeType"],
                    "text": content
                }
            ]
        }

    async def _handle_prompts_list(self) -> dict[str, Any]:
        """Handle prompts list request."""
        return {
            "prompts": list(self.prompts.values())
        }

    async def _handle_prompt_get(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle prompt get request."""
        prompt_name = params.get("name")
        arguments = params.get("arguments", {})

        if prompt_name not in self.prompts:
            raise Exception(f"Prompt '{prompt_name}' not found")

        # Simulate prompt generation
        messages = await self._simulate_prompt_generation(prompt_name, arguments)

        return {
            "description": self.prompts[prompt_name]["description"],
            "messages": messages
        }

    async def _simulate_tool_execution(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Simulate tool execution for demonstration."""
        simulations = {
            "read_file": f"File content from {arguments.get('path', 'unknown')}",
            "write_file": f"Successfully wrote to {arguments.get('path', 'unknown')}",
            "calculate": f"Calculation result: {arguments.get('expression', '0')} = 42",
            "get_weather": f"Weather in {arguments.get('location', 'Unknown')}: 22°C, Sunny"
        }

        return simulations.get(tool_name, f"Executed {tool_name} with {arguments}")

    async def _simulate_resource_content(self, uri: str) -> str:
        """Simulate resource content for demonstration."""
        if uri.endswith(".json"):
            return '{"example": "data", "timestamp": "2024-01-01T00:00:00Z"}'
        elif uri.endswith(".txt"):
            return f"This is the content of {uri}\\nGenerated for demonstration purposes."
        else:
            return f"Binary content of {uri} (simulated)"

    async def _simulate_prompt_generation(self, prompt_name: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """Simulate prompt generation for demonstration."""
        return [
            {
                "role": "system",
                "content": {
                    "type": "text",
                    "text": f"You are using the {prompt_name} prompt template."
                }
            },
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"Generated prompt with arguments: {arguments}"
                }
            }
        ]

    def get_message_log(self) -> list[dict[str, Any]]:
        """Get formatted message log for analysis."""
        log = []
        for i, message in enumerate(self.message_log):
            log.append({
                "sequence": i + 1,
                "type": "request" if message.is_request() else "response" if message.is_response() else "notification",
                "method": message.method,
                "id": message.id,
                "timestamp": datetime.now().isoformat(),
                "message": message.to_json()
            })
        return log


class MockMCPClient:
    """Mock MCP Client for educational demonstrations."""

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.protocol_version = MCPProtocolVersion.V2024_11_05.value
        self.capabilities = MCPCapabilities()
        self.connected_servers: dict[str, MockMCPServer] = {}
        self.message_counter = 0
        self.message_log: list[JSONRPCMessage] = []

        # Set up default capabilities
        self.capabilities.experimental = {}
        self.capabilities.sampling = {}

    def _generate_message_id(self) -> str:
        """Generate unique message ID."""
        self.message_counter += 1
        return f"msg_{self.message_counter}"

    async def connect_to_server(self, server: MockMCPServer) -> bool:
        """Connect to a mock MCP server."""
        try:
            # Send initialization request
            init_request = JSONRPCMessage(
                method="initialize",
                id=self._generate_message_id(),
                params={
                    "protocolVersion": self.protocol_version,
                    "capabilities": self.capabilities.to_dict(),
                    "clientInfo": {
                        "name": self.name,
                        "version": self.version
                    }
                }
            )

            self.message_log.append(init_request)
            response = await server.handle_message(init_request)

            if response and response.result:
                self.connected_servers[server.name] = server
                return True
            else:
                return False

        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    async def list_tools(self, server_name: str) -> dict[str, Any] | None:
        """List tools from a connected server."""
        if server_name not in self.connected_servers:
            return None

        server = self.connected_servers[server_name]

        request = JSONRPCMessage(
            method="tools/list",
            id=self._generate_message_id()
        )

        self.message_log.append(request)
        response = await server.handle_message(request)

        return response.result if response else None

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """Call a tool on a connected server."""
        if server_name not in self.connected_servers:
            return None

        server = self.connected_servers[server_name]

        request = JSONRPCMessage(
            method="tools/call",
            id=self._generate_message_id(),
            params={
                "name": tool_name,
                "arguments": arguments
            }
        )

        self.message_log.append(request)
        response = await server.handle_message(request)

        return response.result if response else None

    async def list_resources(self, server_name: str) -> dict[str, Any] | None:
        """List resources from a connected server."""
        if server_name not in self.connected_servers:
            return None

        server = self.connected_servers[server_name]

        request = JSONRPCMessage(
            method="resources/list",
            id=self._generate_message_id()
        )

        self.message_log.append(request)
        response = await server.handle_message(request)

        return response.result if response else None

    async def read_resource(self, server_name: str, uri: str) -> dict[str, Any] | None:
        """Read a resource from a connected server."""
        if server_name not in self.connected_servers:
            return None

        server = self.connected_servers[server_name]

        request = JSONRPCMessage(
            method="resources/read",
            id=self._generate_message_id(),
            params={"uri": uri}
        )

        self.message_log.append(request)
        response = await server.handle_message(request)

        return response.result if response else None

    def get_message_log(self) -> list[dict[str, Any]]:
        """Get formatted message log for analysis."""
        log = []
        for i, message in enumerate(self.message_log):
            log.append({
                "sequence": i + 1,
                "type": "request" if message.is_request() else "response" if message.is_response() else "notification",
                "method": message.method,
                "id": message.id,
                "timestamp": datetime.now().isoformat(),
                "message": message.to_json()
            })
        return log


class MCPProtocolDemonstrator:
    """Demonstrates MCP protocol interactions with detailed explanations."""

    def __init__(self):
        self.server: MockMCPServer | None = None
        self.client: MockMCPClient | None = None
        self.demo_scenarios: list[dict[str, Any]] = []

    async def setup_demo_environment(self) -> None:
        """Set up a complete demo environment."""
        # Create server
        self.server = MockMCPServer("ProtocolDemoServer")

        # Add sample tools
        self.server.add_tool(
            "echo",
            "Echo back the input text",
            {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to echo"}
                },
                "required": ["text"]
            }
        )

        self.server.add_tool(
            "timestamp",
            "Get current timestamp",
            {
                "type": "object",
                "properties": {},
                "required": []
            }
        )

        # Add sample resources
        self.server.add_resource(
            "demo://config.json",
            "Demo Configuration",
            "Sample configuration resource",
            "application/json"
        )

        # Add sample prompts
        self.server.add_prompt(
            "greeting",
            "Generate a greeting message",
            [
                {"name": "name", "description": "Name to greet", "required": True}
            ]
        )

        # Create client
        self.client = MockMCPClient("ProtocolDemoClient")

    async def demonstrate_initialization_flow(self) -> list[str]:
        """Demonstrate the MCP initialization flow."""
        if not self.server or not self.client:
            await self.setup_demo_environment()

        flow_steps = [
            "🔄 MCP Initialization Flow Demonstration",
            "",
            "Step 1: Client sends initialization request"
        ]

        # Create initialization request
        init_request = JSONRPCMessage(
            method="initialize",
            id="init-1",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "experimental": {},
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "ProtocolDemoClient",
                    "version": "1.0.0"
                }
            }
        )

        flow_steps.extend([
            f"  Request: {init_request.method}",
            f"  ID: {init_request.id}",
            f"  Protocol Version: {init_request.params['protocolVersion']}",
            ""
        ])

        # Process request
        response = await self.server.handle_message(init_request)

        flow_steps.extend([
            "Step 2: Server responds with capabilities",
            f"  Response ID: {response.id}",
            f"  Server Name: {response.result['serverInfo']['name']}",
            f"  Capabilities: {list(response.result['capabilities'].keys())}",
            ""
        ])

        return flow_steps

    async def demonstrate_tool_discovery_and_execution(self) -> list[str]:
        """Demonstrate tool discovery and execution flow."""
        if not self.server or not self.client:
            await self.setup_demo_environment()

        # Connect client to server
        await self.client.connect_to_server(self.server)

        demo_steps = [
            "🔧 Tool Discovery and Execution Flow",
            ""
        ]

        # Step 1: List tools
        tools_result = await self.client.list_tools("ProtocolDemoServer")
        demo_steps.extend([
            "Step 1: Client discovers available tools",
            f"  Found {len(tools_result['tools'])} tools:",
            *[f"    • {tool['name']}: {tool['description']}" for tool in tools_result['tools']],
            ""
        ])

        # Step 2: Call a tool
        echo_result = await self.client.call_tool(
            "ProtocolDemoServer",
            "echo",
            {"text": "Hello, MCP Protocol!"}
        )

        demo_steps.extend([
            "Step 2: Client calls echo tool",
            "  Tool: echo",
            "  Arguments: {'text': 'Hello, MCP Protocol!'}",
            f"  Result: {echo_result['content'][0]['text']}",
            ""
        ])

        # Step 3: Call another tool
        timestamp_result = await self.client.call_tool(
            "ProtocolDemoServer",
            "timestamp",
            {}
        )

        demo_steps.extend([
            "Step 3: Client calls timestamp tool",
            "  Tool: timestamp",
            "  Arguments: (none)",
            f"  Result: {timestamp_result['content'][0]['text']}",
            ""
        ])

        return demo_steps

    async def demonstrate_resource_access_flow(self) -> list[str]:
        """Demonstrate resource access flow."""
        if not self.server or not self.client:
            await self.setup_demo_environment()

        # Connect client to server
        await self.client.connect_to_server(self.server)

        resource_steps = [
            "📄 Resource Access Flow Demonstration",
            ""
        ]

        # Step 1: List resources
        resources_result = await self.client.list_resources("ProtocolDemoServer")
        resource_steps.extend([
            "Step 1: Client discovers available resources",
            f"  Found {len(resources_result['resources'])} resources:",
            *[f"    • {res['uri']}: {res['name']}" for res in resources_result['resources']],
            ""
        ])

        # Step 2: Read a resource
        resource_uri = resources_result['resources'][0]['uri']
        content_result = await self.client.read_resource("ProtocolDemoServer", resource_uri)

        resource_steps.extend([
            "Step 2: Client reads resource content",
            f"  URI: {resource_uri}",
            f"  MIME Type: {content_result['contents'][0]['mimeType']}",
            f"  Content: {content_result['contents'][0]['text'][:100]}...",
            ""
        ])

        return resource_steps

    async def demonstrate_error_scenarios(self) -> list[str]:
        """Demonstrate various error scenarios."""
        if not self.server or not self.client:
            await self.setup_demo_environment()

        # Connect client to server
        await self.client.connect_to_server(self.server)

        error_steps = [
            "🚨 Error Handling Scenarios",
            ""
        ]

        # Scenario 1: Invalid method
        invalid_method_request = JSONRPCMessage(
            method="invalid/method",
            id="error-1"
        )

        error_response = await self.server.handle_message(invalid_method_request)
        error_steps.extend([
            "Scenario 1: Invalid method call",
            f"  Method: {invalid_method_request.method}",
            f"  Error Code: {error_response.error.code}",
            f"  Error Message: {error_response.error.message}",
            ""
        ])

        # Scenario 2: Tool not found
        try:
            await self.client.call_tool("ProtocolDemoServer", "nonexistent_tool", {})
        except Exception:
            error_steps.extend([
                "Scenario 2: Tool not found",
                "  Tool: nonexistent_tool",
                "  Error: Tool not found",
                ""
            ])

        # Scenario 3: Resource not found
        try:
            await self.client.read_resource("ProtocolDemoServer", "demo://nonexistent.txt")
        except Exception:
            error_steps.extend([
                "Scenario 3: Resource not found",
                "  URI: demo://nonexistent.txt",
                "  Error: Resource not found",
                ""
            ])

        return error_steps

    def analyze_message_patterns(self) -> list[str]:
        """Analyze message patterns from the demonstration."""
        if not self.server or not self.client:
            return ["No messages to analyze. Run demonstrations first."]

        server_log = self.server.get_message_log()
        client_log = self.client.get_message_log()

        analysis = [
            "📊 Message Pattern Analysis",
            "",
            f"Total server messages: {len(server_log)}",
            f"Total client messages: {len(client_log)}",
            ""
        ]

        # Analyze message types
        request_count = sum(1 for msg in server_log if msg['type'] == 'request')
        response_count = sum(1 for msg in server_log if msg['type'] == 'response')

        analysis.extend([
            "Message Type Distribution:",
            f"  • Requests: {request_count}",
            f"  • Responses: {response_count}",
            ""
        ])

        # Analyze methods used
        methods = [msg['method'] for msg in server_log if msg['method']]
        unique_methods = set(methods)

        analysis.extend([
            "Methods Used:",
            *[f"  • {method}: {methods.count(method)} times" for method in unique_methods],
            ""
        ])

        return analysis

    def get_protocol_insights(self) -> list[str]:
        """Get key insights about the MCP protocol."""
        return [
            "💡 MCP Protocol Key Insights:",
            "",
            "Protocol Structure:",
            "  • Based on JSON-RPC 2.0 specification",
            "  • Request-response pattern with unique IDs",
            "  • Support for notifications (no response expected)",
            "  • Standardized error codes and messages",
            "",
            "Communication Flow:",
            "  • Initialization establishes capabilities",
            "  • Discovery phase reveals available features",
            "  • Execution phase performs actual work",
            "  • Error handling maintains robustness",
            "",
            "Design Benefits:",
            "  • Language and platform agnostic",
            "  • Extensible through custom methods",
            "  • Robust error handling mechanisms",
            "  • Asynchronous operation support",
            "",
            "Best Practices:",
            "  • Always validate input parameters",
            "  • Provide descriptive error messages",
            "  • Use appropriate HTTP status codes",
            "  • Implement proper timeout handling",
            "  • Log all interactions for debugging"
        ]


# Example usage function
async def run_protocol_demonstration():
    """Run a complete protocol demonstration."""
    demonstrator = MCPProtocolDemonstrator()

    print("Setting up demo environment...")
    await demonstrator.setup_demo_environment()

    print("\\n" + "="*50)
    init_flow = await demonstrator.demonstrate_initialization_flow()
    for line in init_flow:
        print(line)

    print("\\n" + "="*50)
    tool_flow = await demonstrator.demonstrate_tool_discovery_and_execution()
    for line in tool_flow:
        print(line)

    print("\\n" + "="*50)
    resource_flow = await demonstrator.demonstrate_resource_access_flow()
    for line in resource_flow:
        print(line)

    print("\\n" + "="*50)
    error_scenarios = await demonstrator.demonstrate_error_scenarios()
    for line in error_scenarios:
        print(line)

    print("\\n" + "="*50)
    analysis = demonstrator.analyze_message_patterns()
    for line in analysis:
        print(line)

    print("\\n" + "="*50)
    insights = demonstrator.get_protocol_insights()
    for line in insights:
        print(line)


def create_protocol_examples() -> dict[str, dict[str, Any]]:
    """Create a collection of protocol message examples."""
    return {
        "initialization": {
            "request": {
                "jsonrpc": "2.0",
                "id": "1",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"experimental": {}, "sampling": {}},
                    "clientInfo": {"name": "ExampleClient", "version": "1.0.0"}
                }
            },
            "response": {
                "jsonrpc": "2.0",
                "id": "1",
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": True}},
                    "serverInfo": {"name": "ExampleServer", "version": "1.0.0"}
                }
            }
        },
        "tool_call": {
            "request": {
                "jsonrpc": "2.0",
                "id": "2",
                "method": "tools/call",
                "params": {
                    "name": "read_file",
                    "arguments": {"path": "/example.txt"}
                }
            },
            "response": {
                "jsonrpc": "2.0",
                "id": "2",
                "result": {
                    "content": [{"type": "text", "text": "File contents here"}]
                }
            }
        },
        "error_example": {
            "request": {
                "jsonrpc": "2.0",
                "id": "3",
                "method": "invalid/method"
            },
            "response": {
                "jsonrpc": "2.0",
                "id": "3",
                "error": {
                    "code": -32601,
                    "message": "Method not found"
                }
            }
        }
    }
