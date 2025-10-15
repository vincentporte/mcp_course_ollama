"""
Comprehensive demonstration of MCP Client integration with Ollama.

This module demonstrates the complete integration between MCP Clients and Ollama LLMs,
showing how to:
1. Set up the Ollama-MCP bridge
2. Use prompt engineering for effective tool usage
3. Manage conversations with tool integration
4. Handle real-world scenarios and error cases
"""

import asyncio
import logging
from pathlib import Path

from mcp_course.client.basic import BasicMCPClient, ClientConfig
from mcp_course.client.conversation import ConversationManager
from mcp_course.client.integration import MCPToolDefinition, OllamaMCPBridge
from mcp_course.client.prompts import PromptEngineering
from mcp_course.ollama_client.client import OllamaClient
from mcp_course.ollama_client.config import OllamaConfig


async def demonstrate_ollama_mcp_bridge():
    """Demonstrate the core Ollama-MCP bridge functionality."""
    print("=== Ollama-MCP Bridge Demonstration ===\n")

    # 1. Set up the components
    print("1. Setting up MCP Client and Ollama integration...")

    # Configure MCP Client
    client_config = ClientConfig(
        name="integration-demo-client",
        timeout=30.0,
        retry_attempts=2
    )
    mcp_client = BasicMCPClient(client_config)

    # Configure Ollama Client
    ollama_config = OllamaConfig(
        model_name="llama3.2:3b",  # Use a smaller model for demo
        endpoint="http://localhost:11434",
        parameters={
            "temperature": 0.7,
            "max_tokens": 1000
        }
    )
    ollama_client = OllamaClient(ollama_config)

    # Create the bridge
    bridge = OllamaMCPBridge(mcp_client, ollama_client)

    print("✓ Bridge components initialized")

    # 2. Simulate tool discovery (in real usage, this would connect to actual MCP servers)
    print("\n2. Simulating tool discovery...")

    # Add some example tools to demonstrate the integration
    example_tools = [
        MCPToolDefinition(
            name="calculator",
            description="Perform mathematical calculations",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)')"
                    }
                },
                "required": ["expression"]
            },
            server_name="math-server"
        ),
        MCPToolDefinition(
            name="weather_lookup",
            description="Get current weather information for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or location (e.g., 'New York', 'London')"
                    },
                    "units": {
                        "type": "string",
                        "description": "Temperature units",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius"
                    }
                },
                "required": ["location"]
            },
            server_name="weather-server"
        ),
        MCPToolDefinition(
            name="file_search",
            description="Search for files in a directory",
            parameters={
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory path to search in"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "File name pattern to search for (supports wildcards)"
                    }
                },
                "required": ["directory", "pattern"]
            },
            server_name="filesystem-server"
        )
    ]

    # Manually add tools to bridge for demonstration
    for tool in example_tools:
        tool_key = f"{tool.server_name}.{tool.name}"
        bridge.available_tools[tool_key] = tool

    print(f"✓ Discovered {len(example_tools)} tools:")
    for tool in example_tools:
        print(f"  - {tool.name}: {tool.description}")

    # 3. Demonstrate function definitions for Ollama
    print("\n3. Generating Ollama function definitions...")

    ollama_functions = bridge.get_ollama_functions()
    print(f"✓ Generated {len(ollama_functions)} function definitions")

    for func in ollama_functions:
        print(f"  - {func['name']}: {func['description']}")

    # 4. Demonstrate tool-aware prompt creation
    print("\n4. Creating tool-aware prompts...")

    user_message = "I need to calculate the area of a circle with radius 5 and check the weather in Paris"
    tool_aware_prompt = bridge.create_tool_aware_prompt(user_message, include_tool_list=True)

    print("Tool-aware prompt:")
    print("-" * 50)
    print(tool_aware_prompt)
    print("-" * 50)

    return bridge


async def demonstrate_prompt_engineering():
    """Demonstrate advanced prompt engineering for MCP tool usage."""
    print("\n=== Prompt Engineering Demonstration ===\n")

    # Create prompt engineering instance
    prompt_eng = PromptEngineering()

    # Example tools for demonstration
    example_tools = [
        MCPToolDefinition(
            name="data_analyzer",
            description="Analyze data sets and generate insights",
            parameters={
                "type": "object",
                "properties": {
                    "data_source": {"type": "string", "description": "Path to data file"},
                    "analysis_type": {"type": "string", "enum": ["summary", "correlation", "trend"]}
                },
                "required": ["data_source"]
            },
            server_name="analytics-server"
        ),
        MCPToolDefinition(
            name="code_generator",
            description="Generate code snippets based on requirements",
            parameters={
                "type": "object",
                "properties": {
                    "language": {"type": "string", "description": "Programming language"},
                    "requirements": {"type": "string", "description": "Code requirements"}
                },
                "required": ["language", "requirements"]
            },
            server_name="code-server"
        )
    ]

    # 1. Template-based prompt generation
    print("1. Template-based prompt generation...")

    user_message = "Help me analyze sales data and generate a Python script to visualize trends"

    # Try different templates
    templates_to_try = ["direct_tool", "conversational", "step_by_step", "problem_solving"]

    for template_name in templates_to_try:
        print(f"\n--- {template_name.upper()} TEMPLATE ---")
        prompt = prompt_eng.generate_prompt(
            template_name=template_name,
            user_message=user_message,
            tools=example_tools
        )
        # Show first 200 characters
        preview = prompt[:200] + "..." if len(prompt) > 200 else prompt # noqa
        print(preview)

    # 2. Template suggestions
    print("\n2. Automatic template suggestions...")

    test_messages = [
        "How do I use the calculator tool?",
        "I have a problem with my data analysis",
        "Please run the weather lookup for London",
        "Can you help me step by step with this task?"
    ]

    for msg in test_messages:
        suggestions = prompt_eng.get_template_suggestions(msg, example_tools)
        print(f"'{msg}' → Suggested templates: {suggestions}")

    # 3. Model-specific optimization
    print("\n3. Model-specific prompt optimization...")

    base_prompt = "Use the data analyzer tool to examine the sales data and provide comprehensive insights about customer behavior patterns and seasonal trends."

    models_to_test = ["llama3.2:3b", "llama3.2:7b", "codellama:7b"]

    for model in models_to_test:
        optimized = prompt_eng.optimize_prompt_for_model(base_prompt, model, max_tokens=500)
        print(f"\n{model}:")
        print(f"  Original length: {len(base_prompt)} chars")
        print(f"  Optimized length: {len(optimized)} chars")
        print(f"  Preview: {optimized[:100]}...")

    # 4. Context enhancement
    print("\n4. Context-enhanced prompts...")

    # Simulate conversation history
    conversation_history = [
        {"role": "user", "content": "I'm working on a data analysis project"},
        {"role": "assistant", "content": "I'd be happy to help with your data analysis. What kind of data are you working with?"},
        {"role": "user", "content": "Sales data from the last quarter"}
    ]

    # Simulate tool results
    tool_results = {
        "data_analyzer": "Found 1,250 sales records with 15 columns including date, amount, customer_id, and product_category"
    }

    enhanced_prompt = prompt_eng.enhance_prompt_with_context(
        base_prompt=user_message,
        conversation_history=conversation_history,
        tool_results=tool_results,
        user_preferences={"analysis_depth": "detailed", "output_format": "visual"}
    )

    print("Enhanced prompt with context:")
    print("-" * 50)
    print(enhanced_prompt)
    print("-" * 50)

    return prompt_eng


async def demonstrate_conversation_management(bridge: OllamaMCPBridge):
    """Demonstrate conversation management with tool integration."""
    print("\n=== Conversation Management Demonstration ===\n")

    # Create conversation manager
    storage_path = Path.home() / ".mcp_course" / "demo_conversations"
    conv_manager = ConversationManager(bridge, storage_path)

    print("1. Creating and managing conversations...")

    # Create multiple conversations
    conv_ids = []
    conversation_topics = [
        ("Data Analysis Project", ["data", "analysis", "python"]),
        ("Weather Monitoring Setup", ["weather", "monitoring", "api"]),
        ("File Organization Task", ["files", "organization", "automation"])
    ]

    for title, tags in conversation_topics:
        conv_id = conv_manager.create_conversation(title=title, tags=tags)
        conv_ids.append(conv_id)
        print(f"✓ Created conversation: {title} ({conv_id[:8]}...)")

    # 2. Simulate conversation interactions
    print("\n2. Simulating conversation interactions...")

    # Use the first conversation for detailed demonstration
    demo_conv_id = conv_ids[0]

    # Simulate sending messages (in real usage, these would go through Ollama)
    demo_messages = [
        "I need help analyzing sales data from Q3",
        "Can you use the data analyzer tool to examine the trends?",
        "What insights can you provide about customer behavior?"
    ]

    for i, message in enumerate(demo_messages, 1):
        print(f"\nMessage {i}: {message}")

        # In a real scenario, this would call the bridge and get actual responses
        # For demo purposes, we'll simulate the interaction
        try:
            # Simulate the conversation flow
            context = conv_manager.load_conversation(demo_conv_id)
            if context:
                # Add user message to context
                context.messages.append({
                    "role": "user",
                    "content": message,
                    "timestamp": "2024-01-01T12:00:00"
                })

                # Simulate assistant response
                simulated_response = f"I understand you want to {message.lower()}. Let me help you with that using the available tools."
                context.messages.append({
                    "role": "assistant",
                    "content": simulated_response,
                    "timestamp": "2024-01-01T12:00:01"
                })

                # Update conversation
                conv_manager.active_conversations[demo_conv_id] = context
                conv_manager.save_conversation(demo_conv_id)

                print(f"Response: {simulated_response}")

        except Exception as e:
            print(f"Error in conversation simulation: {e}")

    # 3. Conversation search and management
    print("\n3. Conversation search and management...")

    # List all conversations
    all_conversations = conv_manager.list_conversations()
    print(f"✓ Total conversations: {len(all_conversations)}")

    for conv in all_conversations:
        print(f"  - {conv.title} ({conv.message_count} messages, tags: {conv.tags})")

    # Search conversations
    search_results = conv_manager.search_conversations("data", search_content=False)
    print(f"✓ Search results for 'data': {len(search_results)} conversations")

    # Get conversation summary
    summary = conv_manager.get_conversation_summary(demo_conv_id)
    if summary:
        print(f"\nConversation summary for '{summary['title']}':")
        print(f"  - Messages: {summary['message_count']}")
        print(f"  - Tools used: {summary.get('tools_used', [])}")
        print(f"  - Created: {summary['created_at']}")

    # 4. Cleanup demo conversations
    print("\n4. Cleaning up demo conversations...")
    for conv_id in conv_ids:
        conv_manager.delete_conversation(conv_id)
    print("✓ Demo conversations cleaned up")

    return conv_manager


async def demonstrate_error_handling_and_recovery():
    """Demonstrate error handling and recovery patterns."""
    print("\n=== Error Handling and Recovery Demonstration ===\n")

    print("1. Connection error handling...")

    # Test with invalid Ollama endpoint
    invalid_config = OllamaConfig(
        model_name="llama3.2:3b",
        endpoint="http://invalid-endpoint:11434"
    )

    try:
        invalid_client = OllamaClient(invalid_config)
        # This should fail gracefully
        await invalid_client.list_models_async()
        print("Unexpected: Connection succeeded")
    except Exception as e:
        print(f"✓ Handled connection error gracefully: {type(e).__name__}")

    print("\n2. Tool call error handling...")

    # Create a bridge with valid components
    mcp_client = BasicMCPClient()
    ollama_client = OllamaClient(OllamaConfig())
    bridge = OllamaMCPBridge(mcp_client, ollama_client)

    # Try to call a non-existent tool
    result = await bridge.call_mcp_tool("nonexistent_tool", {"arg": "value"})
    if result is None:
        print("✓ Handled missing tool gracefully")

    print("\n3. Conversation recovery...")

    # Test conversation loading with missing files
    conv_manager = ConversationManager(bridge)

    # Try to load non-existent conversation
    missing_conv = conv_manager.load_conversation("nonexistent-id")
    if missing_conv is None:
        print("✓ Handled missing conversation gracefully")

    print("\n4. Prompt generation error handling...")

    prompt_eng = PromptEngineering()

    # Try to use non-existent template
    result = prompt_eng.generate_prompt(
        "nonexistent_template",
        "test message",
        tools=[]
    )
    # Should fallback to original message
    if result == "test message":
        print("✓ Handled missing template gracefully")

    print("\nError handling demonstration completed!")


async def demonstrate_real_world_scenarios():
    """Demonstrate real-world usage scenarios."""
    print("\n=== Real-World Scenarios Demonstration ===\n")

    # Set up components
    mcp_client = BasicMCPClient()
    ollama_client = OllamaClient(OllamaConfig(model_name="llama3.2:3b"))
    bridge = OllamaMCPBridge(mcp_client, ollama_client)
    prompt_eng = PromptEngineering()
    conv_manager = ConversationManager(bridge)

    # Add realistic tools
    realistic_tools = [
        MCPToolDefinition(
            name="database_query",
            description="Execute SQL queries on the company database",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to execute"},
                    "database": {"type": "string", "description": "Database name"}
                },
                "required": ["query"]
            },
            server_name="database-server"
        ),
        MCPToolDefinition(
            name="send_email",
            description="Send email notifications",
            parameters={
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body"}
                },
                "required": ["to", "subject", "body"]
            },
            server_name="email-server"
        ),
        MCPToolDefinition(
            name="create_report",
            description="Generate reports from data",
            parameters={
                "type": "object",
                "properties": {
                    "data_source": {"type": "string", "description": "Data source identifier"},
                    "report_type": {"type": "string", "enum": ["summary", "detailed", "executive"]},
                    "format": {"type": "string", "enum": ["pdf", "html", "csv"]}
                },
                "required": ["data_source", "report_type"]
            },
            server_name="reporting-server"
        )
    ]

    # Add tools to bridge
    for tool in realistic_tools:
        tool_key = f"{tool.server_name}.{tool.name}"
        bridge.available_tools[tool_key] = tool

    # Scenario 1: Business Intelligence Workflow
    print("1. Business Intelligence Workflow...")

    conv_manager.create_conversation(
        title="Monthly Sales Analysis",
        tags=["business", "analytics", "monthly-report"]
    )

    bi_prompt = prompt_eng.generate_prompt(
        "step_by_step",
        "Generate a monthly sales report with insights and email it to the management team",
        tools=realistic_tools
    )

    print("Generated BI workflow prompt:")
    print(bi_prompt[:300] + "...")

    # Scenario 2: Customer Support Automation
    print("\n2. Customer Support Automation...")

    conv_manager.create_conversation(
        title="Customer Issue Resolution",
        tags=["support", "automation", "customer-service"]
    )

    support_prompt = prompt_eng.generate_prompt(
        "problem_solving",
        "A customer reported login issues. Check the database for their account status and send them an update email",
        tools=realistic_tools
    )

    print("Generated support automation prompt:")
    print(support_prompt[:300] + "...")

    # Scenario 3: Data Pipeline Monitoring
    print("\n3. Data Pipeline Monitoring...")

    conv_manager.create_conversation(
        title="Pipeline Health Check",
        tags=["monitoring", "data-pipeline", "alerts"]
    )

    monitoring_prompt = prompt_eng.generate_prompt(
        "direct_tool",
        "Check the status of all data pipelines and create an alert report if any issues are found",
        tools=realistic_tools
    )

    print("Generated monitoring prompt:")
    print(monitoring_prompt[:300] + "...")

    # Show conversation management
    print("\n4. Conversation Management Summary...")

    all_scenarios = conv_manager.list_conversations()
    print(f"Created {len(all_scenarios)} scenario conversations:")

    for conv in all_scenarios:
        print(f"  - {conv.title} (tags: {', '.join(conv.tags)})")

    # Cleanup
    for conv in all_scenarios:
        conv_manager.delete_conversation(conv.id)

    print("\nReal-world scenarios demonstration completed!")


async def main():
    """Run the complete integration demonstration."""
    print("MCP Client + Ollama Integration Demonstration")
    print("=" * 60)

    try:
        # Core integration demonstration
        bridge = await demonstrate_ollama_mcp_bridge()

        # Advanced prompt engineering
        await demonstrate_prompt_engineering()

        # Conversation management
        await demonstrate_conversation_management(bridge)

        # Error handling patterns
        await demonstrate_error_handling_and_recovery()

        # Real-world scenarios
        await demonstrate_real_world_scenarios()

        print("\n" + "=" * 60)
        print("Integration demonstration completed successfully!")
        print("\nKey Integration Points Demonstrated:")
        print("✓ Ollama-MCP Bridge for seamless tool integration")
        print("✓ Advanced prompt engineering for effective tool usage")
        print("✓ Conversation management with persistent context")
        print("✓ Error handling and recovery patterns")
        print("✓ Real-world usage scenarios")

    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user")
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        logging.exception("Integration demonstration error")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run the demonstration
    asyncio.run(main())
