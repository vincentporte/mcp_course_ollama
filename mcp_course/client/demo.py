"""Comprehensive demonstration of the MCP Client framework."""

import asyncio
import logging

from mcp_course.client.basic import BasicMCPClient, ClientConfig
from mcp_course.client.connection import ConnectionManager, ServerDiscovery
from mcp_course.client.conversation import ConversationManager
from mcp_course.client.integration import OllamaMCPBridge
from mcp_course.client.prompts import PromptEngineering
from mcp_course.ollama_client.client import OllamaClient
from mcp_course.ollama_client.config import OllamaConfig


async def demonstrate_complete_mcp_client_framework():
    """
    Comprehensive demonstration of the MCP Client framework.

    This function shows the complete workflow:
    1. Server discovery and configuration
    2. Client connection management
    3. Ollama-MCP integration
    4. Conversation management with tools
    5. Advanced prompt engineering
    """
    print("=== MCP Client Framework Demonstration ===\n")

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # 1. Server Discovery
    print("1. Setting up server discovery...")
    discovery = ServerDiscovery()

    # Add example servers (these would be real MCP servers in practice)
    discovery.add_server(
        name="tutorial-server",
        command="python",
        args=["-m", "mcp_course.server.basic"],
        description="Basic tutorial MCP server",
        tags=["tutorial", "basic"]
    )

    discovery.add_server(
        name="tools-server",
        command="python",
        args=["-m", "mcp_course.server.tools"],
        description="Advanced tools MCP server",
        tags=["tools", "advanced"]
    )

    print(f"Configured {len(discovery.registry.list_servers())} servers")

    # 2. Client Setup
    print("\n2. Setting up MCP Client...")
    client_config = ClientConfig(
        name="demo-client",
        timeout=30.0,
        retry_attempts=3
    )

    client = BasicMCPClient(client_config)

    # Add servers from discovery
    for server_name in discovery.registry.list_servers():
        server_config = discovery.registry.get_server(server_name)
        if server_config:
            await client.add_server(
                name=server_name,
                command=server_config["command"],
                args=server_config["args"],
                env=server_config["env"]
            )

    print(f"Added {len(client.get_all_servers())} servers to client")

    # 3. Connection Management
    print("\n3. Setting up connection management...")
    conn_manager = ConnectionManager(client, max_connections=5)
    await conn_manager.start_health_monitoring()

    # Note: In a real scenario, you would connect to actual servers
    # For this demo, we'll simulate the workflow without actual connections
    print("Connection manager started with health monitoring")

    # 4. Ollama Integration
    print("\n4. Setting up Ollama integration...")
    ollama_config = OllamaConfig(
        model_name="llama3.2:3b",  # Example model
        endpoint="http://localhost:11434"
    )

    ollama_client = OllamaClient(ollama_config)
    bridge = OllamaMCPBridge(client, ollama_client)

    # Simulate tool discovery (would be real tools from connected servers)
    print("Simulating tool discovery...")
    # In practice: tools_discovered = await bridge.discover_tools()
    print("Tool discovery completed (simulated)")

    # 5. Conversation Management
    print("\n5. Setting up conversation management...")
    conv_manager = ConversationManager(bridge)

    # Create a new conversation
    conv_id = conv_manager.create_conversation(
        title="MCP Client Demo Conversation",
        tags=["demo", "tutorial"]
    )

    print(f"Created conversation: {conv_id}")

    # 6. Prompt Engineering
    print("\n6. Demonstrating prompt engineering...")
    prompt_eng = PromptEngineering()

    # Example user message
    user_message = "Help me understand how to use MCP tools effectively"

    # Get template suggestions
    suggestions = prompt_eng.get_template_suggestions(user_message)
    print(f"Suggested prompt templates: {suggestions}")

    # Generate different types of prompts
    for template_name in suggestions[:2]:  # Show first 2 suggestions
        prompt = prompt_eng.generate_prompt(
            template_name=template_name,
            user_message=user_message,
            tools=[]  # Would be real tools in practice
        )
        print(f"\n--- {template_name.upper()} TEMPLATE ---")
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt) # noqa

    # 7. Simulated Conversation Flow
    print("\n7. Simulating conversation flow...")

    # In a real scenario, this would send the message through Ollama with MCP tools
    print("User: Help me understand MCP tools")
    print("Assistant: I'd be happy to help you understand MCP tools! [Simulated response]")

    # Show conversation summary
    summary = conv_manager.get_conversation_summary(conv_id)
    if summary:
        print("\nConversation Summary:")
        print(f"- ID: {summary['id']}")
        print(f"- Title: {summary['title']}")
        print(f"- Messages: {summary['message_count']}")
        print(f"- Tools used: {summary['tools_used']}")

    # 8. Cleanup
    print("\n8. Cleaning up...")
    await conn_manager.shutdown()
    print("Framework demonstration completed!")


async def demonstrate_advanced_features():
    """Demonstrate advanced features of the MCP Client framework."""
    print("\n=== Advanced Features Demonstration ===\n")

    # Advanced prompt engineering
    print("1. Advanced Prompt Engineering...")
    prompt_eng = PromptEngineering()

    # Show all available templates
    templates = prompt_eng.list_templates()
    print("Available prompt templates:")
    for template in templates:
        print(f"  - {template['name']}: {template['description']}")

    # Model-specific optimization
    base_prompt = "Use the calculator tool to compute 2 + 2 and explain the result"
    optimized_prompt = prompt_eng.optimize_prompt_for_model(
        base_prompt,
        "llama3.2:3b",
        max_tokens=1000
    )
    print(f"\nOptimized prompt: {optimized_prompt}")

    # 2. Server validation
    print("\n2. Server Validation...")
    discovery = ServerDiscovery()

    # Add a server for validation
    discovery.add_server(
        name="test-server",
        command="echo",  # Simple command that exists
        args=["hello"],
        description="Test server for validation"
    )

    # Validate server (this will actually try to run the command)
    validation_result = await discovery.validate_server("test-server")
    print(f"Server validation result: {validation_result}")

    # 3. Conversation search and management
    print("\n3. Conversation Management...")

    # Create a mock bridge for conversation manager
    client = BasicMCPClient()
    bridge = OllamaMCPBridge(client)  # Will use default Ollama config
    conv_manager = ConversationManager(bridge)

    # Create multiple conversations
    conv_ids = []
    for i in range(3):
        conv_id = conv_manager.create_conversation(
            title=f"Test Conversation {i+1}",
            tags=["test", f"batch_{i//2}"]
        )
        conv_ids.append(conv_id)

    # List conversations
    conversations = conv_manager.list_conversations(limit=5)
    print(f"Created {len(conversations)} test conversations")

    # Search conversations
    search_results = conv_manager.search_conversations("Test", search_content=False)
    print(f"Search results for 'Test': {len(search_results)} conversations")

    # Clean up test conversations
    for conv_id in conv_ids:
        conv_manager.delete_conversation(conv_id)

    print("Advanced features demonstration completed!")


def demonstrate_educational_patterns():
    """Demonstrate educational patterns for learning MCP Client development."""
    print("\n=== Educational Patterns ===\n")

    print("1. MCP Client Architecture Overview:")
    print("""
    MCP Client Framework Components:

    ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
    │   User Input    │───▶│  Prompt Engine   │───▶│   Ollama LLM    │
    └─────────────────┘    └──────────────────┘    └─────────────────┘
                                    │                        │
                                    ▼                        ▼
    ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
    │ Conversation    │◀───│  MCP Bridge      │───▶│   MCP Client    │
    │ Manager         │    │                  │    │                 │
    └─────────────────┘    └──────────────────┘    └─────────────────┘
                                    │                        │
                                    ▼                        ▼
    ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
    │   Storage       │◀───│  Tool Results    │◀───│  MCP Servers    │
    └─────────────────┘    └──────────────────┘    └─────────────────┘
    """)

    print("2. Key Learning Points:")
    learning_points = [
        "MCP Clients connect to MCP Servers using stdio transport",
        "Tool discovery happens through the list_tools protocol method",
        "Function calling integrates MCP tools with LLM conversations",
        "Conversation management preserves context across multiple turns",
        "Prompt engineering optimizes tool usage for different scenarios",
        "Connection management handles failures and reconnections",
        "Server discovery automates finding and configuring MCP servers"
    ]

    for i, point in enumerate(learning_points, 1):
        print(f"   {i}. {point}")

    print("\n3. Best Practices:")
    best_practices = [
        "Always handle connection failures gracefully",
        "Use appropriate prompt templates for different use cases",
        "Implement proper error handling for tool calls",
        "Save conversation state for continuity",
        "Validate server configurations before connecting",
        "Monitor connection health in production",
        "Optimize prompts for the target LLM model"
    ]

    for i, practice in enumerate(best_practices, 1):
        print(f"   {i}. {practice}")

    print("\n4. Common Patterns:")
    print("""
    # Basic MCP Client Usage Pattern:
    client = BasicMCPClient()
    await client.add_server("my-server", "python", ["-m", "my_mcp_server"])
    await client.connect_to_server("my-server")
    tools = await client.list_server_tools("my-server")
    result = await client.call_tool("my-server", "tool_name", {"arg": "value"})

    # Ollama Integration Pattern:
    bridge = OllamaMCPBridge(client, ollama_client)
    await bridge.discover_tools()
    response = await bridge.chat_with_tools("Use the calculator to compute 2+2")

    # Conversation Management Pattern:
    conv_manager = ConversationManager(bridge)
    conv_id = conv_manager.create_conversation("My Chat")
    result = await conv_manager.send_message(conv_id, "Hello!")
    """)


async def main():
    """Run all demonstrations."""
    try:
        await demonstrate_complete_mcp_client_framework()
        await demonstrate_advanced_features()
        demonstrate_educational_patterns()

    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user")
    except Exception as e:
        print(f"Error during demonstration: {e}")
        logging.exception("Demonstration error")


if __name__ == "__main__":
    asyncio.run(main())
