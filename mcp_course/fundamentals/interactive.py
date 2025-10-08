"""
Interactive protocol flow demonstrations.

This module provides interactive demonstrations of MCP protocol flows,
allowing learners to step through protocol interactions and understand
the communication patterns.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from .protocol import JSONRPCMessage, MockMCPClient, MockMCPServer


class InteractionStep(Enum):
    """Types of interaction steps in MCP protocol flow."""
    INITIALIZATION = "initialization"
    CAPABILITY_DISCOVERY = "capability_discovery"
    TOOL_EXECUTION = "tool_execution"
    RESOURCE_ACCESS = "resource_access"
    ERROR_HANDLING = "error_handling"


@dataclass
class FlowStep:
    """Represents a step in an interactive protocol flow."""
    step_id: str
    title: str
    description: str
    step_type: InteractionStep
    prerequisites: list[str]
    expected_outcome: str
    code_example: str | None = None
    explanation: str | None = None


class InteractiveProtocolFlow:
    """Interactive demonstration of MCP protocol flows."""

    def __init__(self):
        self.current_step = 0
        self.completed_steps: list[str] = []
        self.flow_steps: list[FlowStep] = []
        self.server: MockMCPServer | None = None
        self.client: MockMCPClient | None = None
        self.interaction_log: list[dict[str, Any]] = []

        self._initialize_flow_steps()

    def _initialize_flow_steps(self) -> None:
        """Initialize the predefined flow steps."""
        self.flow_steps = [
            FlowStep(
                step_id="setup",
                title="Environment Setup",
                description="Create and configure MCP server and client instances",
                step_type=InteractionStep.INITIALIZATION,
                prerequisites=[],
                expected_outcome="Server and client instances ready for communication",
                code_example="""
# Create server
server = MockMCPServer("DemoServer")
server.add_tool("echo", "Echo text", {...})

# Create client
client = MockMCPClient("DemoClient")
                """,
                explanation="This step sets up the basic infrastructure needed for MCP communication."
            ),

            FlowStep(
                step_id="initialize",
                title="Protocol Initialization",
                description="Establish connection between client and server",
                step_type=InteractionStep.INITIALIZATION,
                prerequisites=["setup"],
                expected_outcome="Successful handshake with exchanged capabilities",
                code_example="""
# Client sends initialization request
init_request = {
    "jsonrpc": "2.0",
    "id": "1",
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {...},
        "clientInfo": {...}
    }
}
                """,
                explanation="Initialization establishes the protocol version and exchanges capability information."
            ),

            FlowStep(
                step_id="discover_tools",
                title="Tool Discovery",
                description="Client discovers available tools from server",
                step_type=InteractionStep.CAPABILITY_DISCOVERY,
                prerequisites=["initialize"],
                expected_outcome="List of available tools with their schemas",
                code_example="""
# Client requests tool list
tools_request = {
    "jsonrpc": "2.0",
    "id": "2",
    "method": "tools/list"
}
                """,
                explanation="Tool discovery allows the client to understand what capabilities the server provides."
            ),

            FlowStep(
                step_id="call_tool",
                title="Tool Execution",
                description="Client calls a specific tool with parameters",
                step_type=InteractionStep.TOOL_EXECUTION,
                prerequisites=["discover_tools"],
                expected_outcome="Tool execution result or error response",
                code_example="""
# Client calls a tool
tool_call = {
    "jsonrpc": "2.0",
    "id": "3",
    "method": "tools/call",
    "params": {
        "name": "echo",
        "arguments": {"text": "Hello MCP!"}
    }
}
                """,
                explanation="Tool execution is where the actual work happens - the server performs the requested operation."
            ),

            FlowStep(
                step_id="discover_resources",
                title="Resource Discovery",
                description="Client discovers available resources from server",
                step_type=InteractionStep.CAPABILITY_DISCOVERY,
                prerequisites=["initialize"],
                expected_outcome="List of available resources with metadata",
                code_example="""
# Client requests resource list
resources_request = {
    "jsonrpc": "2.0",
    "id": "4",
    "method": "resources/list"
}
                """,
                explanation="Resource discovery reveals what data sources are available through the server."
            ),

            FlowStep(
                step_id="read_resource",
                title="Resource Access",
                description="Client reads content from a specific resource",
                step_type=InteractionStep.RESOURCE_ACCESS,
                prerequisites=["discover_resources"],
                expected_outcome="Resource content or access error",
                code_example="""
# Client reads a resource
resource_read = {
    "jsonrpc": "2.0",
    "id": "5",
    "method": "resources/read",
    "params": {"uri": "file:///example.txt"}
}
                """,
                explanation="Resource access provides the actual data content to the client."
            ),

            FlowStep(
                step_id="handle_errors",
                title="Error Handling",
                description="Demonstrate how errors are handled in the protocol",
                step_type=InteractionStep.ERROR_HANDLING,
                prerequisites=["initialize"],
                expected_outcome="Proper error responses with codes and messages",
                code_example="""
# Invalid method call
invalid_request = {
    "jsonrpc": "2.0",
    "id": "6",
    "method": "invalid/method"
}

# Expected error response
error_response = {
    "jsonrpc": "2.0",
    "id": "6",
    "error": {
        "code": -32601,
        "message": "Method not found"
    }
}
                """,
                explanation="Error handling ensures robust communication even when things go wrong."
            )
        ]

    async def start_interactive_session(self) -> list[str]:
        """Start an interactive protocol demonstration session."""
        session_info = [
            "🎯 Interactive MCP Protocol Flow Session",
            "",
            "This session will guide you through the complete MCP protocol flow.",
            "Each step builds on the previous ones, demonstrating real protocol interactions.",
            "",
            f"Total steps: {len(self.flow_steps)}",
            "Steps overview:",
            *[f"  {i+1}. {step.title}" for i, step in enumerate(self.flow_steps)],
            "",
            "Use next_step() to proceed through the demonstration.",
            "Use get_current_step() to see details of the current step.",
            "Use reset_session() to start over."
        ]

        self.current_step = 0
        self.completed_steps.clear()
        self.interaction_log.clear()

        return session_info

    async def next_step(self) -> tuple[bool, list[str]]:
        """Execute the next step in the protocol flow."""
        if self.current_step >= len(self.flow_steps):
            return False, ["All steps completed! Use reset_session() to start over."]

        step = self.flow_steps[self.current_step]

        # Check prerequisites
        missing_prereqs = [req for req in step.prerequisites if req not in self.completed_steps]
        if missing_prereqs:
            return False, [f"Prerequisites not met: {', '.join(missing_prereqs)}"]

        # Execute the step
        success, output = await self._execute_step(step)

        if success:
            self.completed_steps.append(step.step_id)
            self.current_step += 1

        return success, output

    async def _execute_step(self, step: FlowStep) -> tuple[bool, list[str]]:
        """Execute a specific protocol flow step."""
        output = [
            f"🔄 Executing Step: {step.title}",
            f"Description: {step.description}",
            ""
        ]

        try:
            if step.step_id == "setup":
                success, step_output = await self._execute_setup()
            elif step.step_id == "initialize":
                success, step_output = await self._execute_initialization()
            elif step.step_id == "discover_tools":
                success, step_output = await self._execute_tool_discovery()
            elif step.step_id == "call_tool":
                success, step_output = await self._execute_tool_call()
            elif step.step_id == "discover_resources":
                success, step_output = await self._execute_resource_discovery()
            elif step.step_id == "read_resource":
                success, step_output = await self._execute_resource_read()
            elif step.step_id == "handle_errors":
                success, step_output = await self._execute_error_handling()
            else:
                success, step_output = False, ["Unknown step type"]

            output.extend(step_output)

            if success:
                output.extend([
                    "",
                    "✅ Step completed successfully!",
                    f"Expected outcome: {step.expected_outcome}",
                    ""
                ])

                if step.explanation:
                    output.extend([
                        "💡 Explanation:",
                        f"  {step.explanation}",
                        ""
                    ])

            return success, output

        except Exception as e:
            return False, [f"❌ Step failed: {e!s}"]

    async def _execute_setup(self) -> tuple[bool, list[str]]:
        """Execute environment setup step."""
        self.server = MockMCPServer("InteractiveDemo")

        # Add sample tools
        self.server.add_tool(
            "echo",
            "Echo back input text",
            {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to echo"}
                },
                "required": ["text"]
            }
        )

        self.server.add_tool(
            "calculate",
            "Perform simple calculations",
            {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"]
            }
        )

        # Add sample resources
        self.server.add_resource(
            "demo://config.json",
            "Demo Configuration",
            "Sample configuration data",
            "application/json"
        )

        self.client = MockMCPClient("InteractiveClient")

        return True, [
            "Environment setup completed:",
            f"  • Server created: {self.server.name}",
            f"  • Tools added: {len(self.server.tools)}",
            f"  • Resources added: {len(self.server.resources)}",
            f"  • Client created: {self.client.name}"
        ]

    async def _execute_initialization(self) -> tuple[bool, list[str]]:
        """Execute protocol initialization step."""
        if not self.server or not self.client:
            return False, ["Server or client not set up"]

        success = await self.client.connect_to_server(self.server)

        if success:
            # Log the interaction
            self.interaction_log.append({
                "step": "initialization",
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "details": "Client successfully connected to server"
            })

            return True, [
                "Protocol initialization completed:",
                f"  • Client connected to server: {self.server.name}",
                f"  • Protocol version: {self.client.protocol_version}",
                "  • Capabilities exchanged successfully"
            ]
        else:
            return False, ["Failed to initialize connection"]

    async def _execute_tool_discovery(self) -> tuple[bool, list[str]]:
        """Execute tool discovery step."""
        if not self.client or not self.server:
            return False, ["Client or server not available"]

        tools_result = await self.client.list_tools(self.server.name)

        if tools_result:
            tools = tools_result.get("tools", [])

            self.interaction_log.append({
                "step": "tool_discovery",
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "tools_found": len(tools)
            })

            output = [
                "Tool discovery completed:",
                f"  • Found {len(tools)} tools:"
            ]

            for tool in tools:
                output.append(f"    - {tool['name']}: {tool['description']}")

            return True, output
        else:
            return False, ["Failed to discover tools"]

    async def _execute_tool_call(self) -> tuple[bool, list[str]]:
        """Execute tool call step."""
        if not self.client or not self.server:
            return False, ["Client or server not available"]

        # Call the echo tool
        result = await self.client.call_tool(
            self.server.name,
            "echo",
            {"text": "Hello from interactive demo!"}
        )

        if result:
            self.interaction_log.append({
                "step": "tool_call",
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "tool": "echo",
                "result": result
            })

            return True, [
                "Tool call completed:",
                "  • Called tool: echo",
                "  • Arguments: {'text': 'Hello from interactive demo!'}",
                f"  • Result: {result['content'][0]['text']}"
            ]
        else:
            return False, ["Failed to call tool"]

    async def _execute_resource_discovery(self) -> tuple[bool, list[str]]:
        """Execute resource discovery step."""
        if not self.client or not self.server:
            return False, ["Client or server not available"]

        resources_result = await self.client.list_resources(self.server.name)

        if resources_result:
            resources = resources_result.get("resources", [])

            self.interaction_log.append({
                "step": "resource_discovery",
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "resources_found": len(resources)
            })

            output = [
                "Resource discovery completed:",
                f"  • Found {len(resources)} resources:"
            ]

            for resource in resources:
                output.append(f"    - {resource['uri']}: {resource['name']}")

            return True, output
        else:
            return False, ["Failed to discover resources"]

    async def _execute_resource_read(self) -> tuple[bool, list[str]]:
        """Execute resource read step."""
        if not self.client or not self.server:
            return False, ["Client or server not available"]

        # Read the demo resource
        result = await self.client.read_resource(
            self.server.name,
            "demo://config.json"
        )

        if result:
            content = result['contents'][0]

            self.interaction_log.append({
                "step": "resource_read",
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "uri": "demo://config.json",
                "content_length": len(content['text'])
            })

            return True, [
                "Resource read completed:",
                "  • URI: demo://config.json",
                f"  • MIME Type: {content['mimeType']}",
                f"  • Content: {content['text'][:100]}..."
            ]
        else:
            return False, ["Failed to read resource"]

    async def _execute_error_handling(self) -> tuple[bool, list[str]]:
        """Execute error handling demonstration."""
        if not self.server:
            return False, ["Server not available"]

        # Create an invalid request
        invalid_request = JSONRPCMessage(
            method="invalid/method",
            id="error-demo"
        )

        response = await self.server.handle_message(invalid_request)

        if response and response.error:
            self.interaction_log.append({
                "step": "error_handling",
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "error_code": response.error.code,
                "error_message": response.error.message
            })

            return True, [
                "Error handling demonstrated:",
                f"  • Invalid method: {invalid_request.method}",
                f"  • Error code: {response.error.code}",
                f"  • Error message: {response.error.message}",
                "  • Error properly caught and formatted"
            ]
        else:
            return False, ["Error handling demonstration failed"]

    def get_current_step(self) -> dict[str, Any]:
        """Get details of the current step."""
        if self.current_step >= len(self.flow_steps):
            return {"message": "All steps completed"}

        step = self.flow_steps[self.current_step]
        return {
            "step_number": self.current_step + 1,
            "total_steps": len(self.flow_steps),
            "title": step.title,
            "description": step.description,
            "type": step.step_type.value,
            "prerequisites": step.prerequisites,
            "expected_outcome": step.expected_outcome,
            "code_example": step.code_example,
            "explanation": step.explanation
        }

    def get_session_summary(self) -> list[str]:
        """Get a summary of the current session."""
        summary = [
            "📊 Interactive Session Summary",
            "",
            f"Progress: {len(self.completed_steps)}/{len(self.flow_steps)} steps completed",
            f"Current step: {self.current_step + 1 if self.current_step < len(self.flow_steps) else 'Complete'}",
            ""
        ]

        if self.completed_steps:
            summary.extend([
                "Completed steps:",
                *[f"  ✅ {step_id}" for step_id in self.completed_steps],
                ""
            ])

        if self.interaction_log:
            summary.extend([
                "Interaction log:",
                *[f"  • {log['step']}: {'✅' if log['success'] else '❌'}" for log in self.interaction_log],
                ""
            ])

        return summary

    def reset_session(self) -> list[str]:
        """Reset the interactive session."""
        self.current_step = 0
        self.completed_steps.clear()
        self.interaction_log.clear()
        self.server = None
        self.client = None

        return [
            "🔄 Session reset completed",
            "Ready to start a new interactive demonstration.",
            "Use start_interactive_session() to begin."
        ]


class ProtocolFlowVisualizer:
    """Visualizes MCP protocol flows with step-by-step breakdowns."""

    def __init__(self):
        self.flow_data: dict[str, Any] = {}

    def create_flow_diagram(self, flow_type: str = "complete") -> str:
        """Create a visual flow diagram for MCP protocol interactions."""
        if flow_type == "initialization":
            return self._create_initialization_diagram()
        elif flow_type == "tool_execution":
            return self._create_tool_execution_diagram()
        elif flow_type == "resource_access":
            return self._create_resource_access_diagram()
        else:
            return self._create_complete_flow_diagram()

    def _create_initialization_diagram(self) -> str:
        """Create initialization flow diagram."""
        return """
sequenceDiagram
    participant C as Client
    participant S as Server

    Note over C,S: MCP Initialization Flow

    C->>+S: initialize request
    Note right of C: Protocol version<br/>Client capabilities<br/>Client info

    S-->>-C: initialize response
    Note left of S: Protocol version<br/>Server capabilities<br/>Server info

    Note over C,S: Connection established
        """

    def _create_tool_execution_diagram(self) -> str:
        """Create tool execution flow diagram."""
        return """
sequenceDiagram
    participant C as Client
    participant S as Server
    participant T as Tool

    Note over C,S: Tool Execution Flow

    C->>+S: tools/list request
    S-->>-C: tools/list response
    Note right of S: Available tools<br/>with schemas

    C->>+S: tools/call request
    Note right of C: Tool name<br/>Arguments

    S->>+T: Execute tool
    T-->>-S: Tool result

    S-->>-C: tools/call response
    Note left of S: Execution result<br/>or error
        """

    def _create_resource_access_diagram(self) -> str:
        """Create resource access flow diagram."""
        return """
sequenceDiagram
    participant C as Client
    participant S as Server
    participant R as Resource

    Note over C,S: Resource Access Flow

    C->>+S: resources/list request
    S-->>-C: resources/list response
    Note right of S: Available resources<br/>with metadata

    C->>+S: resources/read request
    Note right of C: Resource URI

    S->>+R: Read resource
    R-->>-S: Resource content

    S-->>-C: resources/read response
    Note left of S: Resource content<br/>or access error
        """

    def _create_complete_flow_diagram(self) -> str:
        """Create complete MCP flow diagram."""
        return """
flowchart TD
    Start([Client Starts]) --> Init[Initialize Connection]
    Init --> InitOK{Initialization OK?}
    InitOK -->|No| Error[Handle Error]
    InitOK -->|Yes| Discover[Discover Capabilities]

    Discover --> Tools[List Tools]
    Discover --> Resources[List Resources]
    Discover --> Prompts[List Prompts]

    Tools --> CallTool[Call Tool]
    Resources --> ReadResource[Read Resource]
    Prompts --> GetPrompt[Get Prompt]

    CallTool --> Process[Process Result]
    ReadResource --> Process
    GetPrompt --> Process

    Process --> More{More Operations?}
    More -->|Yes| Discover
    More -->|No| End([Session Complete])

    Error --> End

    classDef startEnd fill:#e1f5fe
    classDef process fill:#e8f5e8
    classDef decision fill:#fff3e0
    classDef error fill:#ffebee

    class Start,End startEnd
    class Init,Discover,Tools,Resources,Prompts,CallTool,ReadResource,GetPrompt,Process process
    class InitOK,More decision
    class Error error
        """


# Example usage functions
async def run_interactive_demo():
    """Run an interactive protocol demonstration."""
    flow = InteractiveProtocolFlow()

    # Start session
    session_info = await flow.start_interactive_session()
    for line in session_info:
        print(line)

    # Execute all steps
    while flow.current_step < len(flow.flow_steps):
        print(f"\\n{'='*50}")
        success, output = await flow.next_step()
        for line in output:
            print(line)

        if not success:
            break

    # Show summary
    print(f"\\n{'='*50}")
    summary = flow.get_session_summary()
    for line in summary:
        print(line)


def create_interactive_examples() -> dict[str, Any]:
    """Create interactive examples for learning."""
    return {
        "step_by_step": {
            "description": "Step-by-step protocol flow demonstration",
            "usage": """
flow = InteractiveProtocolFlow()
await flow.start_interactive_session()
success, output = await flow.next_step()
            """,
            "benefits": [
                "Learn at your own pace",
                "See actual protocol messages",
                "Understand error handling",
                "Practice with real examples"
            ]
        },
        "visualization": {
            "description": "Visual protocol flow diagrams",
            "usage": """
visualizer = ProtocolFlowVisualizer()
diagram = visualizer.create_flow_diagram("initialization")
            """,
            "benefits": [
                "Visual learning approach",
                "Clear sequence understanding",
                "Component relationship clarity",
                "Flow pattern recognition"
            ]
        }
    }
