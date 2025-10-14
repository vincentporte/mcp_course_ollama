"""
MCP Tools Creation and Registration System

This module provides utilities for creating, validating, and registering
MCP tools with proper parameter validation and response formatting.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
import inspect
from typing import Any

from mcp.types import TextContent, Tool
from pydantic import ValidationError, create_model


@dataclass
class ToolParameter:
    """Represents a parameter for an MCP tool."""

    name: str
    param_type: str
    description: str
    required: bool = True
    enum_values: list[str] | None = None
    default: Any | None = None

    def to_json_schema(self) -> dict[str, Any]:
        """Convert parameter to JSON Schema format."""
        schema = {
            "type": self.param_type,
            "description": self.description
        }

        if self.enum_values:
            schema["enum"] = self.enum_values

        if self.default is not None:
            schema["default"] = self.default

        return schema


@dataclass
class ToolDefinition:
    """Complete definition of an MCP tool."""

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    handler: Callable | None = None

    def to_mcp_tool(self) -> Tool:
        """Convert to MCP Tool format."""
        # Build JSON Schema for parameters
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        input_schema = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False
        }

        return Tool(
            name=self.name,
            description=self.description,
            inputSchema=input_schema
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Validate tool arguments against parameter definitions.

        Args:
            arguments: Raw arguments from tool call

        Returns:
            Validated and type-converted arguments

        Raises:
            ValidationError: If arguments don't match parameter definitions
        """
        # Create Pydantic model for validation
        fields = {}

        for param in self.parameters:
            field_type = self._get_python_type(param.param_type)

            if param.required:
                fields[param.name] = (field_type, ...)
            else:
                fields[param.name] = (field_type, param.default)

        ValidationModel = create_model(f"{self.name}Args", **fields)

        try:
            validated = ValidationModel(**arguments)
            return validated.dict()
        except ValidationError as e:
            raise ValidationError(f"Invalid arguments for tool {self.name}: {e}") from e

    def _get_python_type(self, json_type: str) -> type:
        """Convert JSON Schema type to Python type."""
        type_mapping = {
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict
        }
        return type_mapping.get(json_type, str)


class ToolRegistry:
    """
    Registry for managing MCP tools with validation and execution.

    This class provides a centralized way to register, validate, and execute
    MCP tools with proper error handling and response formatting.
    """

    def __init__(self):
        """Initialize the tool registry."""
        self.tools: dict[str, ToolDefinition] = {}
        self.handlers: dict[str, Callable] = {}

    def register_tool(self, tool_def: ToolDefinition, handler: Callable) -> None:
        """
        Register a tool with its handler function.

        Args:
            tool_def: Complete tool definition
            handler: Async function to handle tool execution
        """
        if not inspect.iscoroutinefunction(handler):
            raise ValueError(f"Tool handler for {tool_def.name} must be an async function")

        self.tools[tool_def.name] = tool_def
        self.handlers[tool_def.name] = handler
        tool_def.handler = handler

    def create_tool(
        self,
        name: str,
        description: str,
        parameters: list[ToolParameter] | None = None
    ) -> ToolDefinition:
        """
        Create a tool definition with fluent interface.

        Args:
            name: Tool name
            description: Tool description
            parameters: List of tool parameters

        Returns:
            ToolDefinition that can be further configured
        """
        return ToolDefinition(
            name=name,
            description=description,
            parameters=parameters or []
        )

    def get_all_tools(self) -> list[Tool]:
        """Get all registered tools in MCP format."""
        return [tool_def.to_mcp_tool() for tool_def in self.tools.values()]

    def get_tool_names(self) -> list[str]:
        """Get names of all registered tools."""
        return list(self.tools.keys())

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self.tools

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """
        Execute a registered tool with validation.

        Args:
            name: Tool name to execute
            arguments: Tool arguments

        Returns:
            List of TextContent responses

        Raises:
            ValueError: If tool not found or validation fails
        """
        if name not in self.tools:
            raise ValueError(f"Tool not found: {name}")

        tool_def = self.tools[name]
        handler = self.handlers[name]

        # Validate arguments
        try:
            validated_args = tool_def.validate_arguments(arguments)
        except ValidationError as e:
            raise ValueError(f"Argument validation failed: {e}") from e

        # Execute handler
        try:
            result = await handler(**validated_args)

            # Ensure result is in correct format
            if isinstance(result, str):
                return [TextContent(type="text", text=result)]
            elif isinstance(result, list) and all(isinstance(item, TextContent) for item in result):
                return result
            elif isinstance(result, TextContent):
                return [result]
            else:
                # Convert other types to string
                return [TextContent(type="text", text=str(result))]

        except Exception as e:
            raise ValueError(f"Tool execution failed: {e}") from e


def create_parameter(
    name: str,
    param_type: str,
    description: str,
    required: bool = True,
    enum_values: list[str] | None = None,
    default: Any | None = None
) -> ToolParameter:
    """
    Convenience function to create tool parameters.

    Args:
        name: Parameter name
        param_type: JSON Schema type (string, number, integer, boolean, array, object)
        description: Parameter description
        required: Whether parameter is required
        enum_values: List of allowed values for enum parameters
        default: Default value if parameter is optional

    Returns:
        ToolParameter instance
    """
    return ToolParameter(
        name=name,
        param_type=param_type,
        description=description,
        required=required,
        enum_values=enum_values,
        default=default
    )


# Decorator for easy tool registration
def tool(
    name: str,
    description: str,
    parameters: list[ToolParameter] | None = None,
    registry: ToolRegistry | None = None
):
    """
    Decorator for registering MCP tools.

    Args:
        name: Tool name
        description: Tool description
        parameters: List of tool parameters
        registry: Tool registry to register with (optional)

    Example:
        @tool("greet", "Greet a person", [
            create_parameter("name", "string", "Person's name")
        ])
        async def greet_person(name: str) -> str:
            return f"Hello, {name}!"
    """
    def decorator(func: Callable):
        tool_def = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters or [],
            handler=func
        )

        if registry:
            registry.register_tool(tool_def, func)

        # Store tool definition on function for later registration
        func._mcp_tool_def = tool_def
        return func

    return decorator


class ToolBuilder:
    """
    Builder class for creating complex tool definitions.

    Provides a fluent interface for building tools with multiple parameters
    and complex validation rules.
    """

    def __init__(self, name: str, description: str):
        """Initialize the tool builder."""
        self.tool_def = ToolDefinition(name=name, description=description)

    def add_parameter(
        self,
        name: str,
        param_type: str,
        description: str,
        required: bool = True,
        enum_values: list[str] | None = None,
        default: Any | None = None
    ) -> "ToolBuilder":
        """Add a parameter to the tool."""
        param = ToolParameter(
            name=name,
            param_type=param_type,
            description=description,
            required=required,
            enum_values=enum_values,
            default=default
        )
        self.tool_def.parameters.append(param)
        return self

    def add_string_param(
        self,
        name: str,
        description: str,
        required: bool = True,
        enum_values: list[str] | None = None,
        default: str | None = None
    ) -> "ToolBuilder":
        """Add a string parameter."""
        return self.add_parameter(name, "string", description, required, enum_values, default)

    def add_number_param(
        self,
        name: str,
        description: str,
        required: bool = True,
        default: float | None = None
    ) -> "ToolBuilder":
        """Add a number parameter."""
        return self.add_parameter(name, "number", description, required, default=default)

    def add_integer_param(
        self,
        name: str,
        description: str,
        required: bool = True,
        default: int | None = None
    ) -> "ToolBuilder":
        """Add an integer parameter."""
        return self.add_parameter(name, "integer", description, required, default=default)

    def add_boolean_param(
        self,
        name: str,
        description: str,
        required: bool = True,
        default: bool | None = None
    ) -> "ToolBuilder":
        """Add a boolean parameter."""
        return self.add_parameter(name, "boolean", description, required, default=default)

    def build(self) -> ToolDefinition:
        """Build the final tool definition."""
        return self.tool_def


def demonstrate_tool_creation():
    """
    Demonstrate various ways to create and register MCP tools.

    This function shows different patterns for tool creation and registration
    that students can use in their own implementations.
    """
    print("=== MCP Tool Creation Examples ===")
    print()

    # Create a tool registry
    registry = ToolRegistry()

    # Example 1: Using ToolBuilder
    print("1. Using ToolBuilder:")
    calculator_tool = (ToolBuilder("calculator", "Perform arithmetic calculations")
                      .add_string_param("operation", "Arithmetic operation", enum_values=["add", "subtract", "multiply", "divide"])
                      .add_number_param("a", "First number")
                      .add_number_param("b", "Second number")
                      .build())

    async def calculator_handler(operation: str, a: float, b: float) -> str:
        if operation == "add":
            return str(a + b)
        elif operation == "subtract":
            return str(a - b)
        elif operation == "multiply":
            return str(a * b)
        elif operation == "divide":
            if b == 0:
                raise ValueError("Division by zero")
            return str(a / b)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    registry.register_tool(calculator_tool, calculator_handler)
    print(f"Created tool: {calculator_tool.name}")
    print(f"Parameters: {[p.name for p in calculator_tool.parameters]}")
    print()

    # Example 2: Using decorator
    print("2. Using decorator:")

    @tool("greet", "Greet a person by name", [
        create_parameter("name", "string", "Person's name"),
        create_parameter("formal", "boolean", "Use formal greeting", required=False, default=False)
    ], registry)
    async def greet_person(name: str, formal: bool = False) -> str:
        if formal:
            return f"Good day, {name}."
        else:
            return f"Hello, {name}!"

    print("Created tool: greet_person")
    print()

    # Example 3: Manual creation
    print("3. Manual creation:")
    weather_tool = ToolDefinition(
        name="get_weather",
        description="Get weather information for a location",
        parameters=[
            ToolParameter("location", "string", "City name or coordinates"),
            ToolParameter("units", "string", "Temperature units", required=False,
                         enum_values=["celsius", "fahrenheit"], default="celsius")
        ]
    )

    async def weather_handler(location: str, units: str = "celsius") -> str:
        # Mock weather data
        temp = 22 if units == "celsius" else 72
        return f"Weather in {location}: {temp}°{'C' if units == 'celsius' else 'F'}, sunny"

    registry.register_tool(weather_tool, weather_handler)
    print(f"Created tool: {weather_tool.name}")
    print()

    # Show all registered tools
    print("All registered tools:")
    for tool_name in registry.get_tool_names():
        print(f"- {tool_name}")

    return registry


if __name__ == "__main__":
    import asyncio

    async def main():
        registry = demonstrate_tool_creation()

        # Test tool execution
        print("\n=== Testing Tool Execution ===")

        try:
            result = await registry.execute_tool("calculator", {
                "operation": "add",
                "a": 5,
                "b": 3
            })
            print(f"Calculator result: {result[0].text}")

            result = await registry.execute_tool("greet", {
                "name": "Alice",
                "formal": True
            })
            print(f"Greeting result: {result[0].text}")

        except Exception as e:
            print(f"Error: {e}")

    asyncio.run(main())
