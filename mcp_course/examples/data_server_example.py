#!/usr/bin/env python3
"""
Data Server Example for Multi-Server Orchestration

This server demonstrates data extraction and management capabilities
for use in multi-server workflows.
"""

import asyncio
from datetime import datetime, timedelta
import json
import random
from typing import Any

from mcp import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from mcp_course.server.scaffolding import ServerConfig


class DataServer:
    """
    MCP Server that provides data extraction and management tools.

    This server demonstrates:
    - Data source simulation
    - Query processing
    - Data formatting for downstream processing
    """

    def __init__(self, config: ServerConfig):
        """Initialize the data server."""
        self.config = config
        self.server = Server(config.name)
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up MCP protocol handlers."""

        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List available data tools."""
            return [
                Tool(
                    name="extract_data",
                    description="Extract data from various sources",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "string",
                                "description": "Data source identifier",
                                "enum": ["sales_database", "user_database", "inventory_database"]
                            },
                            "date_range": {
                                "type": "string",
                                "description": "Date range for data extraction",
                                "enum": ["last_7_days", "last_30_days", "last_90_days", "custom"]
                            },
                            "filters": {
                                "type": "object",
                                "description": "Additional filters to apply",
                                "properties": {
                                    "region": {"type": "string"},
                                    "category": {"type": "string"},
                                    "min_value": {"type": "number"}
                                }
                            }
                        },
                        "required": ["source", "date_range"]
                    }
                ),

                Tool(
                    name="get_schema",
                    description="Get schema information for a data source",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "string",
                                "description": "Data source identifier"
                            }
                        },
                        "required": ["source"]
                    }
                ),

                Tool(
                    name="validate_query",
                    description="Validate a data query before execution",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Query to validate"
                            },
                            "source": {
                                "type": "string",
                                "description": "Target data source"
                            }
                        },
                        "required": ["query", "source"]
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """Handle tool execution."""

            if name == "extract_data":
                return await self._extract_data(arguments)
            elif name == "get_schema":
                return await self._get_schema(arguments)
            elif name == "validate_query":
                return await self._validate_query(arguments)
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

    async def _extract_data(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Extract data from the specified source."""
        source = arguments["source"]
        date_range = arguments["date_range"]
        filters = arguments.get("filters", {})

        # Simulate data extraction based on source
        if source == "sales_database":
            data = self._generate_sales_data(date_range, filters)
        elif source == "user_database":
            data = self._generate_user_data( filters)
        elif source == "inventory_database":
            data = self._generate_inventory_data(filters)
        else:
            return [TextContent(type="text", text=f"Unknown data source: {source}")]

        result = {
            "source": source,
            "date_range": date_range,
            "filters_applied": filters,
            "record_count": len(data),
            "extraction_timestamp": datetime.now().isoformat(),
            "data": data
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _get_schema(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Get schema information for a data source."""
        source = arguments["source"]

        schemas = {
            "sales_database": {
                "table": "sales",
                "columns": [
                    {"name": "id", "type": "integer", "primary_key": True},
                    {"name": "date", "type": "date", "nullable": False},
                    {"name": "product_id", "type": "integer", "nullable": False},
                    {"name": "customer_id", "type": "integer", "nullable": False},
                    {"name": "amount", "type": "decimal", "nullable": False},
                    {"name": "quantity", "type": "integer", "nullable": False},
                    {"name": "region", "type": "string", "nullable": True},
                    {"name": "category", "type": "string", "nullable": True}
                ],
                "indexes": ["date", "customer_id", "product_id"]
            },
            "user_database": {
                "table": "users",
                "columns": [
                    {"name": "id", "type": "integer", "primary_key": True},
                    {"name": "email", "type": "string", "nullable": False, "unique": True},
                    {"name": "name", "type": "string", "nullable": False},
                    {"name": "created_at", "type": "timestamp", "nullable": False},
                    {"name": "last_login", "type": "timestamp", "nullable": True},
                    {"name": "status", "type": "string", "nullable": False},
                    {"name": "region", "type": "string", "nullable": True}
                ],
                "indexes": ["email", "created_at", "status"]
            },
            "inventory_database": {
                "table": "inventory",
                "columns": [
                    {"name": "product_id", "type": "integer", "primary_key": True},
                    {"name": "name", "type": "string", "nullable": False},
                    {"name": "category", "type": "string", "nullable": False},
                    {"name": "stock_level", "type": "integer", "nullable": False},
                    {"name": "price", "type": "decimal", "nullable": False},
                    {"name": "last_updated", "type": "timestamp", "nullable": False}
                ],
                "indexes": ["category", "last_updated"]
            }
        }

        if source in schemas:
            return [TextContent(type="text", text=json.dumps(schemas[source], indent=2))]
        else:
            return [TextContent(type="text", text=f"Schema not found for source: {source}")]

    async def _validate_query(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Validate a data query."""
        query = arguments["query"]
        source = arguments["source"]

        # Simple query validation simulation
        validation_result = {
            "query": query,
            "source": source,
            "valid": True,
            "issues": [],
            "estimated_rows": random.randint(100, 10000),
            "estimated_execution_time": f"{random.uniform(0.1, 5.0):.2f}s"
        }

        # Add some validation logic
        if "DROP" in query.upper():
            validation_result["valid"] = False
            validation_result["issues"].append("DROP statements not allowed")

        if "DELETE" in query.upper():
            validation_result["valid"] = False
            validation_result["issues"].append("DELETE statements not allowed")

        if len(query) > 1000:
            validation_result["issues"].append("Query is very long, consider optimization")

        return [TextContent(type="text", text=json.dumps(validation_result, indent=2))]

    def _generate_sales_data(self, date_range: str, filters: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate sample sales data."""
        # Calculate date range
        end_date = datetime.now()
        if date_range == "last_7_days":
            start_date = end_date - timedelta(days=7)
            num_records = random.randint(50, 200)
        elif date_range == "last_30_days":
            start_date = end_date - timedelta(days=30)
            num_records = random.randint(200, 1000)
        elif date_range == "last_90_days":
            start_date = end_date - timedelta(days=90)
            num_records = random.randint(500, 2000)
        else:
            start_date = end_date - timedelta(days=30)
            num_records = random.randint(100, 500)

        data = []
        regions = ["North", "South", "East", "West", "Central"]
        categories = ["Electronics", "Clothing", "Books", "Home", "Sports"]

        for i in range(num_records):
            # Generate random date within range
            days_diff = (end_date - start_date).days
            random_days = random.randint(0, days_diff)
            sale_date = start_date + timedelta(days=random_days)

            record = {
                "id": i + 1,
                "date": sale_date.strftime("%Y-%m-%d"),
                "product_id": random.randint(1, 1000),
                "customer_id": random.randint(1, 5000),
                "amount": round(random.uniform(10.0, 500.0), 2),
                "quantity": random.randint(1, 10),
                "region": random.choice(regions),
                "category": random.choice(categories)
            }

            # Apply filters
            if filters.get("region") and record["region"] != filters["region"]:
                continue
            if filters.get("category") and record["category"] != filters["category"]:
                continue
            if filters.get("min_value") and record["amount"] < filters["min_value"]:
                continue

            data.append(record)

        return data

    def _generate_user_data(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate sample user data."""
        num_records = random.randint(100, 1000)
        data = []

        regions = ["North", "South", "East", "West", "Central"]
        statuses = ["active", "inactive", "pending", "suspended"]

        for i in range(num_records):
            record = {
                "id": i + 1,
                "email": f"user{i+1}@example.com",
                "name": f"User {i+1}",
                "created_at": (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
                "last_login": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
                "status": random.choice(statuses),
                "region": random.choice(regions)
            }

            # Apply filters
            if filters.get("region") and record["region"] != filters["region"]:
                continue

            data.append(record)

        return data

    def _generate_inventory_data(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate sample inventory data."""
        num_records = random.randint(50, 500)
        data = []

        categories = ["Electronics", "Clothing", "Books", "Home", "Sports"]

        for i in range(num_records):
            record = {
                "product_id": i + 1,
                "name": f"Product {i+1}",
                "category": random.choice(categories),
                "stock_level": random.randint(0, 1000),
                "price": round(random.uniform(5.0, 200.0), 2),
                "last_updated": datetime.now().isoformat()
            }

            # Apply filters
            if filters.get("category") and record["category"] != filters["category"]:
                continue

            data.append(record)

        return data

    async def run(self):
        """Run the data server."""
        async with stdio_server() as (read_stream, write_stream):
            init_options = InitializationOptions(
                server_name=self.config.name,
                server_version=self.config.version,
                capabilities=self.config.capabilities
            )
            await self.server.run(read_stream, write_stream, init_options)


async def main():
    """Main entry point."""
    config = ServerConfig(
        name="data-server",
        version="1.0.0",
        description="Data extraction and management server for multi-server workflows"
    )

    server = DataServer(config)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
