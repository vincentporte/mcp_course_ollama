"""
MCP Server Implementation Tutorial

This module provides step-by-step tutorials for understanding and implementing
MCP Servers, covering fundamental concepts and practical examples.
"""
from dataclasses import dataclass
import json

from mcp_course.server.basic import demonstrate_request_response_patterns


@dataclass
class TutorialStep:
    """Represents a single step in the MCP Server tutorial."""

    title: str
    description: str
    code_example: str
    explanation: str
    key_concepts: list[str]


class MCPServerTutorial:
    """
    Interactive tutorial for learning MCP Server implementation.

    This class provides a structured approach to learning MCP Server
    development through progressive examples and explanations.
    """

    def __init__(self):
        """Initialize the tutorial with all steps."""
        self.steps = self._create_tutorial_steps()
        self.current_step = 0

    def _create_tutorial_steps(self) -> list[TutorialStep]:
        """Create the complete tutorial step sequence."""
        return [
            TutorialStep(
                title="Understanding MCP Server Basics",
                description="Learn what an MCP Server is and its role in the MCP ecosystem",
                code_example='''
# MCP Server is a component that:
# 1. Exposes tools (functions) that LLMs can call
# 2. Provides resources (data) that LLMs can access
# 3. Offers prompts (templates) for common tasks
# 4. Communicates via JSON-RPC protocol

from mcp import server

# Create a server instance
server = Server("my-first-server")
                ''',
                explanation="""
An MCP Server acts as a bridge between Large Language Models (LLMs) and external
functionality. It exposes capabilities through three main types:

- **Tools**: Functions that the LLM can call to perform actions
- **Resources**: Data sources that the LLM can read from
- **Prompts**: Pre-defined prompt templates for common tasks

The server communicates with MCP Clients (like LLMs) using the JSON-RPC protocol
over various transports (stdio, HTTP, WebSocket).
                """,
                key_concepts=[
                    "MCP Server role and purpose",
                    "Tools, Resources, and Prompts",
                    "JSON-RPC communication protocol",
                    "Server initialization"
                ]
            ),

            TutorialStep(
                title="Server Configuration and Setup",
                description="Learn how to properly configure an MCP Server",
                code_example='''
from mcp_course.server.scaffolding import ServerConfig, create_server_scaffold

# Create server configuration
config = ServerConfig(
    name="tutorial-server",
    version="1.0.0",
    description="My first MCP Server",
    capabilities={
        "tools": {},      # Will support tools
        "resources": {},  # Will support resources
        "prompts": {}     # Will support prompts
    },
    logging_level="INFO"
)

# Create server with configuration
server = create_server_scaffold(config)
                ''',
                explanation="""
Proper server configuration is crucial for MCP Server functionality:

- **Name**: Unique identifier for your server
- **Version**: Semantic version for compatibility tracking
- **Description**: Human-readable description of server purpose
- **Capabilities**: Declares what types of functionality the server provides
- **Logging**: Essential for debugging and monitoring

The ServerConfig class provides a structured way to manage these settings
and ensures consistent server initialization.
                """,
                key_concepts=[
                    "Server configuration patterns",
                    "Capability declaration",
                    "Version management",
                    "Logging setup"
                ]
            ),

            TutorialStep(
                title="Request/Response Handling",
                description="Understand how MCP Servers handle client requests",
                code_example='''
@server.list_tools()
async def handle_list_tools():
    """Handle requests for available tools."""
    return [
        Tool(
            name="greet",
            description="Greet a person by name",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                },
                "required": ["name"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict):
    """Handle tool execution requests."""
    if name == "greet":
        person_name = arguments["name"]
        return [TextContent(type="text", text=f"Hello, {person_name}!")]
                ''',
                explanation="""
MCP Servers use decorators to register request handlers:

- **@server.list_tools()**: Handles requests for available tools
- **@server.call_tool()**: Handles tool execution requests
- **@server.list_resources()**: Handles resource listing requests
- **@server.read_resource()**: Handles resource content requests

Each handler is an async function that receives structured parameters
and returns typed responses. The MCP library handles JSON-RPC serialization
and protocol details automatically.
                """,
                key_concepts=[
                    "Handler registration with decorators",
                    "Async request processing",
                    "Structured input/output",
                    "JSON-RPC abstraction"
                ]
            ),

            TutorialStep(
                title="Running the Server",
                description="Learn how to start and manage an MCP Server",
                code_example='''
import asyncio
from mcp.server.stdio import stdio_server
from mcp.server.models import InitializationOptions

async def run_server():
    """Run the MCP Server with stdio transport."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="tutorial-server",
                server_version="1.0.0",
                capabilities={"tools": {}, "resources": {}, "prompts": {}}
            )
        )

# Start the server
if __name__ == "__main__":
    asyncio.run(run_server())
                ''',
                explanation="""
MCP Servers typically run with stdio transport for integration with MCP Clients:

- **stdio_server()**: Creates read/write streams for stdio communication
- **server.run()**: Starts the server event loop
- **InitializationOptions**: Provides server metadata to clients
- **asyncio.run()**: Manages the async event loop

The server runs indefinitely, processing requests from MCP Clients until
the connection is closed or an error occurs.
                """,
                key_concepts=[
                    "Stdio transport setup",
                    "Server lifecycle management",
                    "Async event loop handling",
                    "Initialization options"
                ]
            ),

            TutorialStep(
                title="Error Handling and Logging",
                description="Implement robust error handling and logging",
                code_example='''
import logging

# Set up logging
logger = logging.getLogger("tutorial-server")

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict):
    """Handle tool calls with proper error handling."""
    try:
        logger.info(f"Executing tool: {name}")

        if name == "divide":
            a = arguments["a"]
            b = arguments["b"]

            if b == 0:
                raise ValueError("Division by zero not allowed")

            result = a / b
            logger.info(f"Division result: {result}")
            return [TextContent(type="text", text=str(result))]

        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]
                ''',
                explanation="""
Robust error handling is essential for production MCP Servers:

- **Logging**: Use structured logging to track server operations
- **Exception Handling**: Catch and handle errors gracefully
- **Error Messages**: Provide clear, actionable error messages
- **Validation**: Validate inputs before processing

Good error handling improves debugging, user experience, and server reliability.
The MCP protocol allows servers to return error information in responses.
                """,
                key_concepts=[
                    "Structured logging practices",
                    "Exception handling patterns",
                    "Input validation",
                    "Error response formatting"
                ]
            )
        ]

    def get_step(self, step_number: int) -> TutorialStep | None:
        """Get a specific tutorial step."""
        if 0 <= step_number < len(self.steps):
            return self.steps[step_number]
        return None

    def next_step(self) -> TutorialStep | None:
        """Move to the next tutorial step."""
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            return self.steps[self.current_step]
        return None

    def previous_step(self) -> TutorialStep | None:
        """Move to the previous tutorial step."""
        if self.current_step > 0:
            self.current_step -= 1
            return self.steps[self.current_step]
        return None

    def get_current_step(self) -> TutorialStep:
        """Get the current tutorial step."""
        return self.steps[self.current_step]

    def get_progress(self) -> tuple[int, int]:
        """Get current progress as (current_step, total_steps)."""
        return (self.current_step + 1, len(self.steps))

    def get_all_concepts(self) -> list[str]:
        """Get all key concepts covered in the tutorial."""
        concepts = []
        for step in self.steps:
            concepts.extend(step.key_concepts)
        return concepts


def demonstrate_protocol_messages():
    """
    Demonstrate the JSON-RPC messages used in MCP communication.

    This function shows the actual protocol messages that flow between
    MCP Clients and Servers during typical interactions.
    """
    print("=== MCP Protocol Message Examples ===")
    print()

    examples = demonstrate_request_response_patterns()

    for message_type, message_data in examples.items():
        print(f"{message_type.replace('_', ' ').title()}:")
        print("-" * 40)
        print(json.dumps(message_data, indent=2))
        print()


def run_interactive_tutorial():
    """
    Run an interactive tutorial session.

    This function provides a command-line interface for working through
    the MCP Server tutorial step by step.
    """
    tutorial = MCPServerTutorial()

    print("=== Interactive MCP Server Tutorial ===")
    print("Learn how to build MCP Servers step by step!")
    print()

    while True:
        step = tutorial.get_current_step()
        progress = tutorial.get_progress()

        print(f"Step {progress[0]}/{progress[1]}: {step.title}")
        print("=" * 50)
        print(step.description)
        print()

        print("Code Example:")
        print("-" * 20)
        print(step.code_example)
        print()

        print("Explanation:")
        print("-" * 20)
        print(step.explanation.strip())
        print()

        print("Key Concepts:")
        print("-" * 20)
        for concept in step.key_concepts:
            print(f"• {concept}")
        print()

        # Simple navigation
        print("Commands: (n)ext, (p)revious, (q)uit")
        choice = input("Your choice: ").lower().strip()

        if choice == 'q':
            break
        elif choice == 'n':
            if not tutorial.next_step():
                print("Tutorial complete! 🎉")
                break
        elif choice == 'p':
            tutorial.previous_step()

        print("\n" + "="*60 + "\n")

    print("Thank you for completing the MCP Server tutorial!")


if __name__ == "__main__":
    run_interactive_tutorial()
