"""
Test suite for MCP Client integration with Ollama.

This module provides comprehensive tests for the integration components:
- OllamaMCPBridge functionality
- Prompt engineering utilities
- Conversation management
- Error handling and edge cases
"""

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import AsyncMock, Mock

import pytest

from mcp_course.client.basic import BasicMCPClient
from mcp_course.client.conversation import ConversationManager
from mcp_course.client.integration import MCPToolDefinition, OllamaMCPBridge
from mcp_course.client.prompts import PromptEngineering
from mcp_course.ollama_client.client import OllamaClient
from mcp_course.ollama_client.config import OllamaConfig


class TestMCPToolDefinition:
    """Test MCPToolDefinition functionality."""

    def test_tool_definition_creation(self):
        """Test creating a tool definition."""
        tool = MCPToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            server_name="test-server"
        )

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.server_name == "test-server"

    def test_to_ollama_function(self):
        """Test converting tool definition to Ollama function format."""
        tool = MCPToolDefinition(
            name="calculator",
            description="Perform calculations",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"]
            },
            server_name="math-server"
        )

        ollama_func = tool.to_ollama_function()

        assert ollama_func["name"] == "calculator"
        assert ollama_func["description"] == "Perform calculations"
        assert "expression" in ollama_func["parameters"]["properties"]


class TestOllamaMCPBridge:
    """Test OllamaMCPBridge functionality."""

    @pytest.fixture
    def mock_mcp_client(self):
        """Create a mock MCP client."""
        client = Mock(spec=BasicMCPClient)
        client.get_connected_servers.return_value = ["test-server"]
        return client

    @pytest.fixture
    def mock_ollama_client(self):
        """Create a mock Ollama client."""
        config = OllamaConfig(model_name="test-model")
        client = Mock(spec=OllamaClient)
        client.config = config
        return client

    @pytest.fixture
    def bridge(self, mock_mcp_client, mock_ollama_client):
        """Create a bridge instance for testing."""
        return OllamaMCPBridge(mock_mcp_client, mock_ollama_client)

    def test_bridge_initialization(self, bridge):
        """Test bridge initialization."""
        assert bridge.mcp_client is not None
        assert bridge.ollama_client is not None
        assert isinstance(bridge.available_tools, dict)
        assert len(bridge.available_tools) == 0

    @pytest.mark.asyncio
    async def test_discover_tools_empty(self, bridge, mock_mcp_client):
        """Test tool discovery with no tools."""
        mock_mcp_client.list_server_tools = AsyncMock(return_value=[])

        count = await bridge.discover_tools()

        assert count == 0
        assert len(bridge.available_tools) == 0

    @pytest.mark.asyncio
    async def test_discover_tools_with_tools(self, bridge, mock_mcp_client):
        """Test tool discovery with actual tools."""
        # Mock tool object
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool description"
        mock_tool.inputSchema = {"type": "object", "properties": {}}

        mock_mcp_client.list_server_tools = AsyncMock(return_value=[mock_tool])

        count = await bridge.discover_tools()

        assert count == 1
        assert "test-server.test_tool" in bridge.available_tools

        tool_def = bridge.available_tools["test-server.test_tool"]
        assert tool_def.name == "test_tool"
        assert tool_def.server_name == "test-server"

    def test_get_ollama_functions(self, bridge):
        """Test getting Ollama function definitions."""
        # Add a test tool
        tool = MCPToolDefinition(
            name="test_func",
            description="Test function",
            parameters={"type": "object"},
            server_name="test-server"
        )
        bridge.available_tools["test-server.test_func"] = tool

        functions = bridge.get_ollama_functions()

        assert len(functions) == 1
        assert functions[0]["name"] == "test_func"
        assert functions[0]["description"] == "Test function"

    @pytest.mark.asyncio
    async def test_call_mcp_tool_not_found(self, bridge):
        """Test calling a non-existent tool."""
        result = await bridge.call_mcp_tool("nonexistent", {})

        assert result is None

    @pytest.mark.asyncio
    async def test_call_mcp_tool_success(self, bridge, mock_mcp_client):
        """Test successful tool call."""
        # Add a test tool
        tool = MCPToolDefinition(
            name="test_tool",
            description="Test tool",
            parameters={"type": "object"},
            server_name="test-server"
        )
        bridge.available_tools["test-server.test_tool"] = tool

        # Mock successful tool call
        mock_result = Mock()
        mock_mcp_client.call_tool = AsyncMock(return_value=mock_result)

        result = await bridge.call_mcp_tool("test_tool", {"arg": "value"})

        assert result == mock_result
        mock_mcp_client.call_tool.assert_called_once_with(
            "test-server", "test_tool", {"arg": "value"}
        )

    def test_create_tool_aware_prompt(self, bridge):
        """Test creating tool-aware prompts."""
        # Add a test tool
        tool = MCPToolDefinition(
            name="calculator",
            description="Perform calculations",
            parameters={"type": "object"},
            server_name="math-server"
        )
        bridge.available_tools["math-server.calculator"] = tool

        prompt = bridge.create_tool_aware_prompt("Calculate 2+2", include_tool_list=True)

        assert "calculator" in prompt
        assert "Perform calculations" in prompt
        assert "Calculate 2+2" in prompt

    def test_create_tool_aware_prompt_no_tools(self, bridge):
        """Test creating prompt with no tools available."""
        prompt = bridge.create_tool_aware_prompt("Hello", include_tool_list=True)

        assert prompt == "Hello"


class TestPromptEngineering:
    """Test PromptEngineering functionality."""

    @pytest.fixture
    def prompt_eng(self):
        """Create a PromptEngineering instance."""
        return PromptEngineering()

    @pytest.fixture
    def sample_tools(self):
        """Create sample tools for testing."""
        return [
            MCPToolDefinition(
                name="calculator",
                description="Perform mathematical calculations",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression"}
                    },
                    "required": ["expression"]
                },
                server_name="math-server"
            ),
            MCPToolDefinition(
                name="weather",
                description="Get weather information",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "Location name"}
                    },
                    "required": ["location"]
                },
                server_name="weather-server"
            )
        ]

    def test_initialization(self, prompt_eng):
        """Test PromptEngineering initialization."""
        assert len(prompt_eng.templates) > 0
        assert "direct_tool" in prompt_eng.templates
        assert "conversational" in prompt_eng.templates

    def test_generate_tool_list_simple(self, prompt_eng, sample_tools):
        """Test generating simple tool list."""
        tool_list = prompt_eng.generate_tool_list(sample_tools, "simple")

        assert "calculator" in tool_list
        assert "weather" in tool_list
        assert "Perform mathematical calculations" in tool_list

    def test_generate_tool_list_detailed(self, prompt_eng, sample_tools):
        """Test generating detailed tool list."""
        tool_list = prompt_eng.generate_tool_list(sample_tools, "detailed")

        assert "**calculator**" in tool_list
        assert "math-server" in tool_list
        assert "Parameters:" in tool_list
        assert "expression" in tool_list

    def test_generate_tool_list_json(self, prompt_eng, sample_tools):
        """Test generating JSON tool list."""
        tool_list = prompt_eng.generate_tool_list(sample_tools, "json")

        # Should be valid JSON
        parsed = json.loads(tool_list)
        assert len(parsed) == 2
        assert parsed[0]["name"] == "calculator"

    def test_generate_prompt_direct_tool(self, prompt_eng, sample_tools):
        """Test generating direct tool prompt."""
        prompt = prompt_eng.generate_prompt(
            "direct_tool",
            "Calculate 2+2",
            tools=sample_tools
        )

        assert "Calculate 2+2" in prompt
        assert "calculator" in prompt
        assert "weather" in prompt

    def test_generate_prompt_nonexistent_template(self, prompt_eng):
        """Test generating prompt with non-existent template."""
        prompt = prompt_eng.generate_prompt(
            "nonexistent_template",
            "Test message",
            tools=[]
        )

        # Should fallback to original message
        assert prompt == "Test message"

    def test_get_template_suggestions(self, prompt_eng, sample_tools):
        """Test getting template suggestions."""
        # Test different message types
        test_cases = [
            ("How do I use this tool?", ["step_by_step"]),
            ("I have a problem", ["problem_solving"]),
            ("Please run the calculator", ["direct_tool"]),
            ("Hello there", ["conversational"])
        ]

        for message, expected_templates in test_cases:
            suggestions = prompt_eng.get_template_suggestions(message, sample_tools)
            for expected in expected_templates:
                assert expected in suggestions

    def test_optimize_prompt_for_model(self, prompt_eng):
        """Test model-specific prompt optimization."""
        base_prompt = "Please utilize the appropriate tools to provide a comprehensive response"

        # Test optimization for smaller model
        optimized = prompt_eng.optimize_prompt_for_model(base_prompt, "llama3.2:3b")

        # Should simplify language for smaller models
        assert "utilize" not in optimized or "use" in optimized

    def test_enhance_prompt_with_context(self, prompt_eng):
        """Test enhancing prompt with context."""
        base_prompt = "Calculate something"

        conversation_history = [
            {"role": "user", "content": "I need help with math"},
            {"role": "assistant", "content": "I can help with calculations"}
        ]

        tool_results = {
            "calculator": "Previous calculation: 2+2=4"
        }

        enhanced = prompt_eng.enhance_prompt_with_context(
            base_prompt,
            conversation_history=conversation_history,
            tool_results=tool_results
        )

        assert "Calculate something" in enhanced
        assert "I need help with math" in enhanced
        assert "Previous calculation" in enhanced

    def test_list_templates(self, prompt_eng):
        """Test listing available templates."""
        templates = prompt_eng.list_templates()

        assert len(templates) > 0
        assert all("name" in template for template in templates)
        assert all("description" in template for template in templates)


class TestConversationManager:
    """Test ConversationManager functionality."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def mock_bridge(self):
        """Create a mock bridge."""
        return Mock(spec=OllamaMCPBridge)

    @pytest.fixture
    def conv_manager(self, mock_bridge, temp_storage):
        """Create a ConversationManager instance."""
        return ConversationManager(mock_bridge, temp_storage)

    def test_initialization(self, conv_manager):
        """Test ConversationManager initialization."""
        assert conv_manager.bridge is not None
        assert conv_manager.storage_path.exists()
        assert isinstance(conv_manager.active_conversations, dict)
        assert isinstance(conv_manager.conversation_metadata, dict)

    def test_create_conversation(self, conv_manager):
        """Test creating a new conversation."""
        conv_id = conv_manager.create_conversation(
            title="Test Conversation",
            tags=["test", "demo"]
        )

        assert conv_id in conv_manager.active_conversations
        assert conv_id in conv_manager.conversation_metadata

        metadata = conv_manager.conversation_metadata[conv_id]
        assert metadata.title == "Test Conversation"
        assert "test" in metadata.tags
        assert "demo" in metadata.tags

    def test_save_and_load_conversation(self, conv_manager):
        """Test saving and loading conversations."""
        # Create a conversation
        conv_id = conv_manager.create_conversation("Test Save/Load")

        # Add some messages
        context = conv_manager.active_conversations[conv_id]
        context.messages.append({
            "role": "user",
            "content": "Hello",
            "timestamp": datetime.now().isoformat()
        })

        # Save the conversation
        success = conv_manager.save_conversation(conv_id)
        assert success

        # Remove from memory
        del conv_manager.active_conversations[conv_id]

        # Load it back
        loaded_context = conv_manager.load_conversation(conv_id)
        assert loaded_context is not None
        assert len(loaded_context.messages) == 1
        assert loaded_context.messages[0]["content"] == "Hello"

    def test_list_conversations(self, conv_manager):
        """Test listing conversations."""
        # Create multiple conversations
        conv_ids = []
        for i in range(3):
            conv_id = conv_manager.create_conversation(
                title=f"Conversation {i}",
                tags=["test", f"batch_{i//2}"]
            )
            conv_ids.append(conv_id)

        # List all conversations
        all_convs = conv_manager.list_conversations()
        assert len(all_convs) == 3

        # List with tag filter
        tagged_convs = conv_manager.list_conversations(tag="batch_0")
        assert len(tagged_convs) == 2

        # List with limit
        limited_convs = conv_manager.list_conversations(limit=2)
        assert len(limited_convs) == 2

    def test_delete_conversation(self, conv_manager):
        """Test deleting conversations."""
        # Create a conversation
        conv_id = conv_manager.create_conversation("To Delete")

        # Verify it exists
        assert conv_id in conv_manager.conversation_metadata

        # Delete it
        success = conv_manager.delete_conversation(conv_id)
        assert success

        # Verify it's gone
        assert conv_id not in conv_manager.conversation_metadata
        assert conv_id not in conv_manager.active_conversations

    def test_search_conversations(self, conv_manager):
        """Test searching conversations."""
        # Create conversations with different titles
        conv_manager.create_conversation("Data Analysis Project", ["data"])
        conv_manager.create_conversation("Weather Monitoring", ["weather"])
        conv_manager.create_conversation("Data Processing", ["data"])

        # Search by title
        results = conv_manager.search_conversations("Data", search_content=False)
        assert len(results) == 2

        # Search by tag (should be found in title search too)
        weather_results = conv_manager.search_conversations("weather", search_content=False)
        assert len(weather_results) == 1

    def test_get_conversation_summary(self, conv_manager):
        """Test getting conversation summary."""
        # Create a conversation
        conv_id = conv_manager.create_conversation(
            "Summary Test",
            tags=["summary", "test"]
        )

        # Add some messages
        context = conv_manager.active_conversations[conv_id]
        context.messages.extend([
            {"role": "user", "content": "Hello", "timestamp": "2024-01-01T12:00:00"},
            {"role": "assistant", "content": "Hi there!", "timestamp": "2024-01-01T12:00:01"}
        ])
        context.tool_results["calculator"] = "2+2=4"

        # Get summary
        summary = conv_manager.get_conversation_summary(conv_id)

        assert summary is not None
        assert summary["title"] == "Summary Test"
        assert "summary" in summary["tags"]
        assert "calculator" in summary["tools_used"]
        assert len(summary["recent_messages"]) == 2


class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    @pytest.fixture
    def integration_setup(self):
        """Set up complete integration environment."""
        # Create mock components
        mcp_client = Mock(spec=BasicMCPClient)
        mcp_client.get_connected_servers.return_value = ["test-server"]

        ollama_config = OllamaConfig(model_name="test-model")
        ollama_client = Mock(spec=OllamaClient)
        ollama_client.config = ollama_config

        # Create bridge
        bridge = OllamaMCPBridge(mcp_client, ollama_client)

        # Add test tools
        test_tool = MCPToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            server_name="test-server"
        )
        bridge.available_tools["test-server.test_tool"] = test_tool

        # Create other components
        prompt_eng = PromptEngineering()

        with tempfile.TemporaryDirectory() as temp_dir:
            conv_manager = ConversationManager(bridge, Path(temp_dir))

            yield {
                "bridge": bridge,
                "prompt_eng": prompt_eng,
                "conv_manager": conv_manager,
                "mcp_client": mcp_client,
                "ollama_client": ollama_client
            }

    def test_end_to_end_workflow(self, integration_setup):
        """Test complete end-to-end workflow."""
        bridge = integration_setup["bridge"]
        prompt_eng = integration_setup["prompt_eng"]
        conv_manager = integration_setup["conv_manager"]

        # 1. Create conversation
        conv_id = conv_manager.create_conversation(
            "E2E Test Workflow",
            tags=["test", "e2e"]
        )

        # 2. Generate tool-aware prompt
        user_message = "Use the test tool to help me"
        tools = list(bridge.available_tools.values())

        prompt = prompt_eng.generate_prompt(
            "direct_tool",
            user_message,
            tools=tools
        )

        assert "test_tool" in prompt
        assert user_message in prompt

        # 3. Simulate conversation interaction
        context = conv_manager.load_conversation(conv_id)
        context.messages.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })

        # 4. Save conversation
        success = conv_manager.save_conversation(conv_id)
        assert success

        # 5. Verify conversation state
        summary = conv_manager.get_conversation_summary(conv_id)
        assert summary["title"] == "E2E Test Workflow"
        assert summary["message_count"] == 1


def run_integration_tests():
    """Run all integration tests."""
    print("Running MCP Client Integration Tests...")

    # Run pytest programmatically
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        __file__,
        "-v",
        "--tb=short"
    ], check=False, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return result.returncode == 0


if __name__ == "__main__":
    success = run_integration_tests()
    if success:
        print("✓ All integration tests passed!")
    else:
        print("✗ Some tests failed!")
        sys.exit(1)
