#!/usr/bin/env python3
"""
Advanced MCP Tools Example

This example demonstrates advanced patterns for creating MCP tools including:
- Complex parameter validation
- Tool composition and chaining
- Error handling and recovery
- Response formatting options

Run this example:
    python -m mcp_course.examples.advanced_tools_example
"""

import asyncio
from datetime import datetime
import json

from mcp_course.server.tools import (
    ToolBuilder,
    ToolRegistry,
    create_parameter,
    tool,
)


class AdvancedToolsExample:
    """Demonstrates advanced MCP tool patterns."""

    def __init__(self):
        """Initialize the advanced tools example."""
        self.registry = ToolRegistry()
        self._setup_tools()

    def _setup_tools(self):
        """Set up all example tools."""

        # File operations tool with complex validation
        file_ops_tool = (ToolBuilder("file_operations", "Perform file system operations")
                        .add_string_param("operation", "File operation to perform",
                                        enum_values=["create", "read", "update", "delete", "list"])
                        .add_string_param("path", "File or directory path")
                        .add_string_param("content", "File content (for create/update)", required=False)
                        .add_boolean_param("recursive", "Recursive operation", required=False, default=False)
                        .build())

        async def file_operations_handler(
            operation: str,
            path: str,
            content: str | None = None,
            recursive: bool = False
        ) -> str:
            """Handle file operations with validation."""
            # Simulate file operations (in real implementation, use actual file I/O)
            if operation == "create":
                if not content:
                    raise ValueError("Content required for create operation")
                return f"Created file {path} with {len(content)} characters"

            elif operation == "read":
                return f"Reading file {path}: [simulated content]"

            elif operation == "update":
                if not content:
                    raise ValueError("Content required for update operation")
                return f"Updated file {path} with {len(content)} characters"

            elif operation == "delete":
                return f"Deleted {'recursively ' if recursive else ''}{path}"

            elif operation == "list":
                return f"Listing {'recursive ' if recursive else ''}contents of {path}"

            else:
                raise ValueError(f"Unknown operation: {operation}")

        self.registry.register_tool(file_ops_tool, file_operations_handler)

        # Data processing tool with array parameters
        data_tool = (ToolBuilder("process_data", "Process and analyze data arrays")
                    .add_parameter("data", "array", "Array of numbers to process")
                    .add_string_param("operation", "Processing operation",
                                    enum_values=["sum", "average", "min", "max", "sort", "filter"])
                    .add_number_param("threshold", "Threshold value for filtering", required=False)
                    .build())

        async def process_data_handler(
            data: list[float],
            operation: str,
            threshold: float | None = None
        ) -> str:
            """Process data arrays with various operations."""
            if not data:
                raise ValueError("Data array cannot be empty")

            if operation == "sum":
                result = sum(data)
            elif operation == "average":
                result = sum(data) / len(data)
            elif operation == "min":
                result = min(data)
            elif operation == "max":
                result = max(data)
            elif operation == "sort":
                result = sorted(data)
                return f"Sorted data: {result}"
            elif operation == "filter":
                if threshold is None:
                    raise ValueError("Threshold required for filter operation")
                filtered = [x for x in data if x > threshold]
                return f"Filtered data (> {threshold}): {filtered}"
            else:
                raise ValueError(f"Unknown operation: {operation}")

            return f"{operation.title()} of {data}: {result}"

        self.registry.register_tool(data_tool, process_data_handler)

        # API simulation tool with object parameters
        api_tool = (ToolBuilder("api_request", "Simulate API requests with various methods")
                   .add_string_param("method", "HTTP method", enum_values=["GET", "POST", "PUT", "DELETE"])
                   .add_string_param("url", "API endpoint URL")
                   .add_parameter("headers", "object", "Request headers", required=False)
                   .add_parameter("body", "object", "Request body (for POST/PUT)", required=False)
                   .add_integer_param("timeout", "Request timeout in seconds", required=False, default=30)
                   .build())

        async def api_request_handler(
            method: str,
            url: str,
            headers: dict | None = None,
            body: dict | None = None,
            timeout: int = 30
        ) -> str:
            """Simulate API requests with proper validation."""
            # Validate method-specific requirements
            if method in ["POST", "PUT"] and body is None:
                raise ValueError(f"{method} requests require a body")

            # Simulate API call
            response = {
                "method": method,
                "url": url,
                "status": 200,
                "headers": headers or {},
                "timeout": timeout,
                "timestamp": datetime.now().isoformat()
            }

            if body:
                response["request_body"] = body

            return f"API {method} {url}: {json.dumps(response, indent=2)}"

        self.registry.register_tool(api_tool, api_request_handler)

        # Tool with complex business logic
        @tool("calculate_loan", "Calculate loan payments and amortization", [
            create_parameter("principal", "number", "Loan principal amount"),
            create_parameter("rate", "number", "Annual interest rate (as decimal, e.g., 0.05 for 5%)"),
            create_parameter("years", "integer", "Loan term in years"),
            create_parameter("payment_frequency", "string", "Payment frequency",
                           enum_values=["monthly", "quarterly", "annually"], default="monthly")
        ], self.registry)
        async def calculate_loan(
            principal: float,
            rate: float,
            years: int,
            payment_frequency: str = "monthly"
        ) -> str:
            """Calculate loan payments with detailed breakdown."""
            if principal <= 0:
                raise ValueError("Principal must be positive")
            if rate < 0:
                raise ValueError("Interest rate cannot be negative")
            if years <= 0:
                raise ValueError("Loan term must be positive")

            # Calculate payment frequency
            freq_map = {"monthly": 12, "quarterly": 4, "annually": 1}
            payments_per_year = freq_map[payment_frequency]
            total_payments = years * payments_per_year
            period_rate = rate / payments_per_year

            # Calculate payment amount using amortization formula
            if rate == 0:
                payment = principal / total_payments
            else:
                payment = principal * (period_rate * (1 + period_rate) ** total_payments) / \
                         ((1 + period_rate) ** total_payments - 1)

            total_paid = payment * total_payments
            total_interest = total_paid - principal

            result = {
                "principal": principal,
                "annual_rate": rate * 100,
                "term_years": years,
                "payment_frequency": payment_frequency,
                "payment_amount": round(payment, 2),
                "total_payments": total_payments,
                "total_paid": round(total_paid, 2),
                "total_interest": round(total_interest, 2),
                "interest_percentage": round((total_interest / principal) * 100, 2)
            }

            return f"Loan Calculation Results:\n{json.dumps(result, indent=2)}"

    async def demonstrate_tools(self):
        """Demonstrate all the advanced tools."""
        print("=== Advanced MCP Tools Demonstration ===")
        print()

        # Show all available tools
        print("Available Tools:")
        for tool_name in self.registry.get_tool_names():
            print(f"- {tool_name}")
        print()

        # Test file operations
        print("1. File Operations Tool:")
        try:
            result = await self.registry.execute_tool("file_operations", {
                "operation": "create",
                "path": "/tmp/test.txt",
                "content": "Hello, MCP World!"
            })
            print(f"   {result[0].text}")
        except Exception as e:
            print(f"   Error: {e}")
        print()

        # Test data processing
        print("2. Data Processing Tool:")
        try:
            result = await self.registry.execute_tool("process_data", {
                "data": [1, 5, 3, 9, 2, 7],
                "operation": "filter",
                "threshold": 4
            })
            print(f"   {result[0].text}")
        except Exception as e:
            print(f"   Error: {e}")
        print()

        # Test API simulation
        print("3. API Request Tool:")
        try:
            result = await self.registry.execute_tool("api_request", {
                "method": "POST",
                "url": "https://api.example.com/users",
                "headers": {"Content-Type": "application/json"},
                "body": {"name": "John Doe", "email": "john@example.com"}
            })
            print(f"   {result[0].text}")
        except Exception as e:
            print(f"   Error: {e}")
        print()

        # Test loan calculator
        print("4. Loan Calculator Tool:")
        try:
            result = await self.registry.execute_tool("calculate_loan", {
                "principal": 250000,
                "rate": 0.045,
                "years": 30,
                "payment_frequency": "monthly"
            })
            print(f"   {result[0].text}")
        except Exception as e:
            print(f"   Error: {e}")
        print()

    async def demonstrate_error_handling(self):
        """Demonstrate error handling in tools."""
        print("=== Error Handling Demonstration ===")
        print()

        # Test validation errors
        test_cases = [
            ("file_operations", {"operation": "create", "path": "/tmp/test.txt"}),  # Missing content
            ("process_data", {"data": [], "operation": "sum"}),  # Empty data
            ("api_request", {"method": "POST", "url": "https://example.com"}),  # Missing body
            ("calculate_loan", {"principal": -1000, "rate": 0.05, "years": 30}),  # Negative principal
        ]

        for tool_name, args in test_cases:
            try:
                await self.registry.execute_tool(tool_name, args)
                print(f"❌ Expected error for {tool_name} with {args}")
            except Exception as e:
                print(f"✅ Caught expected error for {tool_name}: {e}")
        print()


async def main():
    """Run the advanced tools example."""
    example = AdvancedToolsExample()

    await example.demonstrate_tools()
    await example.demonstrate_error_handling()

    print("Advanced tools demonstration complete!")


if __name__ == "__main__":
    asyncio.run(main())
