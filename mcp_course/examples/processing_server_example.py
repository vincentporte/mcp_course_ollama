#!/usr/bin/env python3
"""
Processing Server Example for Multi-Server Orchestration

This server demonstrates data processing and transformation capabilities
for use in multi-server workflows.
"""

import asyncio
from datetime import datetime
import json
import statistics
from typing import Any

from mcp import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from mcp_course.server.scaffolding import ServerConfig


class ProcessingServer:
    """
    MCP Server that provides data processing and transformation tools.

    This server demonstrates:
    - Data cleaning and normalization
    - Statistical processing
    - Data transformation and formatting
    - Report generation
    """

    def __init__(self, config: ServerConfig):
        """Initialize the processing server."""
        self.config = config
        self.server = Server(config.name)
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up MCP protocol handlers."""

        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List available processing tools."""
            return [
                Tool(
                    name="clean_data",
                    description="Clean and normalize data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "string",
                                "description": "JSON string containing data to clean"
                            },
                            "remove_outliers": {
                                "type": "boolean",
                                "description": "Whether to remove statistical outliers",
                                "default": False
                            },
                            "normalize_fields": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Fields to normalize"
                            },
                            "fill_missing": {
                                "type": "string",
                                "enum": ["mean", "median", "zero", "remove"],
                                "description": "How to handle missing values",
                                "default": "remove"
                            }
                        },
                        "required": ["data"]
                    }
                ),

                Tool(
                    name="transform_data",
                    description="Transform data structure and format",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "string",
                                "description": "JSON string containing data to transform"
                            },
                            "transformation": {
                                "type": "string",
                                "enum": ["pivot", "aggregate", "flatten", "group_by"],
                                "description": "Type of transformation to apply"
                            },
                            "parameters": {
                                "type": "object",
                                "description": "Transformation-specific parameters"
                            }
                        },
                        "required": ["data", "transformation"]
                    }
                ),

                Tool(
                    name="generate_report",
                    description="Generate formatted reports from processed data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "statistics": {
                                "type": "string",
                                "description": "JSON string containing statistical data"
                            },
                            "trends": {
                                "type": "string",
                                "description": "JSON string containing trend analysis"
                            },
                            "format": {
                                "type": "string",
                                "enum": ["json", "markdown", "html", "csv"],
                                "description": "Output format for the report",
                                "default": "json"
                            },
                            "include_charts": {
                                "type": "boolean",
                                "description": "Whether to include chart descriptions",
                                "default": False
                            }
                        },
                        "required": ["statistics", "trends"]
                    }
                ),

                Tool(
                    name="validate_data_quality",
                    description="Validate data quality and completeness",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "string",
                                "description": "JSON string containing data to validate"
                            },
                            "schema": {
                                "type": "object",
                                "description": "Expected data schema for validation"
                            },
                            "quality_checks": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of quality checks to perform"
                            }
                        },
                        "required": ["data"]
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """Handle tool execution."""

            if name == "clean_data":
                return await self._clean_data(arguments)
            elif name == "transform_data":
                return await self._transform_data(arguments)
            elif name == "generate_report":
                return await self._generate_report(arguments)
            elif name == "validate_data_quality":
                return await self._validate_data_quality(arguments)
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

    async def _clean_data(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Clean and normalize data."""
        try:
            # Parse input data
            data_str = arguments["data"]
            data_obj = json.loads(data_str) if isinstance(data_str, str) else data_str

            # Extract the actual data array
            if isinstance(data_obj, dict) and "data" in data_obj:
                data_records = data_obj["data"]
                metadata = {k: v for k, v in data_obj.items() if k != "data"}
            else:
                data_records = data_obj
                metadata = {}

            remove_outliers = arguments.get("remove_outliers", False)
            normalize_fields = arguments.get("normalize_fields", [])
            fill_missing = arguments.get("fill_missing", "remove")

            cleaned_data = []

            for record in data_records:
                if not isinstance(record, dict):
                    continue

                cleaned_record = record.copy()

                # Handle missing values
                if fill_missing == "remove":
                    # Skip records with missing required fields
                    if any(v is None or v == "" for v in cleaned_record.values()):
                        continue
                elif fill_missing == "zero":
                    # Fill numeric missing values with zero
                    for key, value in cleaned_record.items():
                        if value is None or (value == "" and key in ["amount", "quantity", "price", "stock_level"]):
                            cleaned_record[key] = 0

                # Normalize specified fields
                for field in normalize_fields:
                    if field in cleaned_record and isinstance(cleaned_record[field], str):
                        cleaned_record[field] = cleaned_record[field].strip().lower()

                cleaned_data.append(cleaned_record)

            # Remove outliers if requested
            if remove_outliers and cleaned_data:
                cleaned_data = self._remove_outliers(cleaned_data)

            result = {
                "original_count": len(data_records),
                "cleaned_count": len(cleaned_data),
                "removed_count": len(data_records) - len(cleaned_data),
                "cleaning_timestamp": datetime.now().isoformat(),
                "cleaning_parameters": {
                    "remove_outliers": remove_outliers,
                    "normalize_fields": normalize_fields,
                    "fill_missing": fill_missing
                },
                "data": cleaned_data,
                **metadata
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Error cleaning data: {e!s}")]

    async def _transform_data(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Transform data structure and format."""
        try:
            data_str = arguments["data"]
            data_obj = json.loads(data_str) if isinstance(data_str, str) else data_str

            # Extract the actual data array
            data_records = data_obj["data"] if isinstance(data_obj, dict) and "data" in data_obj else data_obj

            transformation = arguments["transformation"]
            parameters = arguments.get("parameters", {})

            if transformation == "aggregate":
                result = self._aggregate_data(data_records, parameters)
            elif transformation == "group_by":
                result = self._group_by_data(data_records, parameters)
            elif transformation == "pivot":
                result = self._pivot_data()
            elif transformation == "flatten":
                result = self._flatten_data(data_records)
            else:
                return [TextContent(type="text", text=f"Unknown transformation: {transformation}")]

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Error transforming data: {e!s}")]

    async def _generate_report(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Generate formatted reports from processed data."""
        try:
            statistics_str = arguments["statistics"]
            trends_str = arguments["trends"]
            format_type = arguments.get("format", "json")
            include_charts = arguments.get("include_charts", False)

            # Parse input data
            statistics_data = json.loads(statistics_str) if isinstance(statistics_str, str) else statistics_str

            trends_data = json.loads(trends_str) if isinstance(trends_str, str) else trends_str

            # Generate report based on format
            if format_type == "json":
                report = self._generate_json_report(statistics_data, trends_data, include_charts)
            elif format_type == "markdown":
                report = self._generate_markdown_report(statistics_data, trends_data, include_charts)
            elif format_type == "html":
                report = self._generate_html_report(statistics_data, trends_data)
            else:
                report = self._generate_json_report(statistics_data, trends_data, include_charts)

            return [TextContent(type="text", text=report)]

        except Exception as e:
            return [TextContent(type="text", text=f"Error generating report: {e!s}")]

    async def _validate_data_quality(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Validate data quality and completeness."""
        try:
            data_str = arguments["data"]
            data_obj = json.loads(data_str) if isinstance(data_str, str) else data_str

            # Extract the actual data array
            data_records = data_obj["data"] if isinstance(data_obj, dict) and "data" in data_obj else data_obj

            arguments.get("schema", {})
            quality_checks = arguments.get("quality_checks", ["completeness", "consistency", "validity"])

            validation_result = {
                "total_records": len(data_records),
                "validation_timestamp": datetime.now().isoformat(),
                "checks_performed": quality_checks,
                "issues": [],
                "quality_score": 0.0,
                "recommendations": []
            }

            # Perform quality checks
            if "completeness" in quality_checks:
                completeness_issues = self._check_completeness(data_records)
                validation_result["issues"].extend(completeness_issues)

            if "consistency" in quality_checks:
                consistency_issues = self._check_consistency(data_records)
                validation_result["issues"].extend(consistency_issues)

            if "validity" in quality_checks:
                validity_issues = self._check_validity(data_records)
                validation_result["issues"].extend(validity_issues)

            # Calculate quality score
            total_possible_issues = len(data_records) * len(quality_checks)
            actual_issues = len(validation_result["issues"])
            validation_result["quality_score"] = max(0.0, 1.0 - (actual_issues / max(1, total_possible_issues)))

            # Generate recommendations
            if validation_result["quality_score"] < 0.8:
                validation_result["recommendations"].append("Consider data cleaning before processing")
            if actual_issues > 0:
                validation_result["recommendations"].append("Review and fix identified data quality issues")

            return [TextContent(type="text", text=json.dumps(validation_result, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Error validating data quality: {e!s}")]

    def _remove_outliers(self, data_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove statistical outliers from numeric fields."""
        if not data_records:
            return data_records

        # Find numeric fields
        numeric_fields = []
        for record in data_records[:10]:  # Sample first 10 records
            for key, value in record.items():
                if isinstance(value, (int, float)) and key not in numeric_fields:
                    numeric_fields.append(key)

        # Calculate outlier thresholds for each numeric field
        outlier_thresholds = {}
        for field in numeric_fields:
            values = [record.get(field, 0) for record in data_records if isinstance(record.get(field), (int, float))]
            if len(values) > 3:
                q1 = statistics.quantiles(values, n=4)[0]
                q3 = statistics.quantiles(values, n=4)[2]
                iqr = q3 - q1
                outlier_thresholds[field] = {
                    "lower": q1 - 1.5 * iqr,
                    "upper": q3 + 1.5 * iqr
                }

        # Filter out outliers
        filtered_data = []
        for record in data_records:
            is_outlier = False
            for field, thresholds in outlier_thresholds.items():
                value = record.get(field)
                if isinstance(value, (int, float)) and (value < thresholds["lower"] or value > thresholds["upper"]):
                    is_outlier = True
                    break

            if not is_outlier:
                filtered_data.append(record)

        return filtered_data

    def _aggregate_data(self, data_records: list[dict[str, Any]], parameters: dict[str, Any]) -> dict[str, Any]:
        """Aggregate data by specified fields."""
        group_by = parameters.get("group_by", [])
        aggregations = parameters.get("aggregations", {"count": "count"})

        if not group_by:
            # Global aggregation
            result = {"total_records": len(data_records)}
            for agg_name, agg_field in aggregations.items():
                if agg_field == "count":
                    result[agg_name] = len(data_records)
                elif agg_field in data_records[0] if data_records else {}:
                    values = [record.get(agg_field, 0) for record in data_records if isinstance(record.get(agg_field), (int, float))]
                    if values:
                        if agg_name.startswith("sum"):
                            result[agg_name] = sum(values)
                        elif agg_name.startswith("avg") or agg_name.startswith("mean"):
                            result[agg_name] = statistics.mean(values)
                        elif agg_name.startswith("max"):
                            result[agg_name] = max(values)
                        elif agg_name.startswith("min"):
                            result[agg_name] = min(values)
            return result

        # Group by aggregation
        groups = {}
        for record in data_records:
            key = tuple(record.get(field, "unknown") for field in group_by)
            if key not in groups:
                groups[key] = []
            groups[key].append(record)

        result = {}
        for key, group_records in groups.items():
            group_key = "_".join(str(k) for k in key)
            group_result = {"count": len(group_records)}

            for agg_name, agg_field in aggregations.items():
                if agg_field != "count" and agg_field in group_records[0] if group_records else {}:
                    values = [record.get(agg_field, 0) for record in group_records if isinstance(record.get(agg_field), (int, float))]
                    if values:
                        if agg_name.startswith("sum"):
                            group_result[agg_name] = sum(values)
                        elif agg_name.startswith("avg") or agg_name.startswith("mean"):
                            group_result[agg_name] = statistics.mean(values)

            result[group_key] = group_result

        return result

    def _group_by_data(self, data_records: list[dict[str, Any]], parameters: dict[str, Any]) -> dict[str, Any]:
        """Group data by specified fields."""
        group_by_field = parameters.get("field", "category")

        groups = {}
        for record in data_records:
            key = record.get(group_by_field, "unknown")
            if key not in groups:
                groups[key] = []
            groups[key].append(record)

        return {
            "grouped_by": group_by_field,
            "group_count": len(groups),
            "groups": {k: {"count": len(v), "records": v} for k, v in groups.items()}
        }

    def _pivot_data(self) -> dict[str, Any]:
        """Pivot data structure."""
        # Simple pivot implementation
        return {"message": "Pivot transformation not fully implemented in this example"}

    def _flatten_data(self, data_records: list[dict[str, Any]]) -> dict[str, Any]:
        """Flatten nested data structures."""
        flattened = []
        for record in data_records:
            flat_record = {}
            for key, value in record.items():
                if isinstance(value, dict):
                    for nested_key, nested_value in value.items():
                        flat_record[f"{key}_{nested_key}"] = nested_value
                else:
                    flat_record[key] = value
            flattened.append(flat_record)

        return {"flattened_records": flattened}

    def _generate_json_report(self, statistics_data: dict[str, Any], trends_data: dict[str, Any], include_charts: bool) -> str:
        """Generate JSON format report."""
        report = {
            "report_type": "comprehensive_analysis",
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "statistics_summary": self._summarize_statistics(statistics_data),
                "trends_summary": self._summarize_trends(trends_data)
            },
            "detailed_statistics": statistics_data,
            "detailed_trends": trends_data
        }

        if include_charts:
            report["chart_recommendations"] = [
                "Bar chart for category distribution",
                "Line chart for trend analysis over time",
                "Pie chart for regional breakdown"
            ]

        return json.dumps(report, indent=2)

    def _generate_markdown_report(self, statistics_data: dict[str, Any], trends_data: dict[str, Any], include_charts: bool) -> str:
        """Generate Markdown format report."""
        report = f"""# Comprehensive Data Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

{self._summarize_statistics(statistics_data)}

{self._summarize_trends(trends_data)}

## Detailed Statistics

```json
{json.dumps(statistics_data, indent=2)}
```

## Trend Analysis

```json
{json.dumps(trends_data, indent=2)}
```
"""

        if include_charts:
            report += """
## Recommended Visualizations

- Bar chart for category distribution
- Line chart for trend analysis over time
- Pie chart for regional breakdown
"""

        return report

    def _generate_html_report(self, statistics_data: dict[str, Any], trends_data: dict[str, Any]) -> str:
        """Generate HTML format report."""
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Data Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .summary {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
        .data {{ background-color: #f9f9f9; padding: 15px; border-left: 4px solid #007acc; }}
    </style>
</head>
<body>
    <h1>Comprehensive Data Analysis Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="summary">
        <h2>Executive Summary</h2>
        <p>{self._summarize_statistics(statistics_data)}</p>
        <p>{self._summarize_trends(trends_data)}</p>
    </div>

    <h2>Detailed Analysis</h2>
    <div class="data">
        <pre>{json.dumps({**statistics_data, **trends_data}, indent=2)}</pre>
    </div>
</body>
</html>"""

    def _summarize_statistics(self, statistics_data: dict[str, Any]) -> str:
        """Generate a summary of statistics data."""
        return f"Statistical analysis completed with {len(statistics_data)} metrics calculated."

    def _summarize_trends(self, trends_data: dict[str, Any]) -> str:
        """Generate a summary of trends data."""
        return f"Trend analysis identified {len(trends_data)} key patterns in the data."

    def _check_completeness(self, data_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Check data completeness."""
        issues = []
        for i, record in enumerate(data_records):
            missing_fields = [k for k, v in record.items() if v is None or v == ""]
            if missing_fields:
                issues.append({
                    "type": "completeness",
                    "record_index": i,
                    "description": f"Missing values in fields: {missing_fields}"
                })
        return issues

    def _check_consistency(self, data_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Check data consistency."""
        issues = []
        # Simple consistency check - look for data type inconsistencies
        if data_records:
            field_types = {}
            for field in data_records[0]:
                field_types[field] = type(data_records[0][field])

            for i, record in enumerate(data_records[1:], 1):
                for field, expected_type in field_types.items():
                    if field in record and type(record[field]) is not expected_type:
                        issues.append({
                            "type": "consistency",
                            "record_index": i,
                            "description": f"Type mismatch in field {field}: expected {expected_type.__name__}, got {type(record[field]).__name__}"
                        })
        return issues

    def _check_validity(self, data_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Check data validity against schema."""
        issues = []
        # Simple validity check based on basic constraints
        for i, record in enumerate(data_records):
            # Check for negative values where they shouldn't be
            for field, value in record.items():
                if field in ["amount", "quantity", "price", "stock_level"] and isinstance(value, (int, float)) and value < 0:
                    issues.append({
                        "type": "validity",
                        "record_index": i,
                        "description": f"Negative value in field {field}: {value}"
                    })
        return issues

    async def run(self):
        """Run the processing server."""
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
        name="processing-server",
        version="1.0.0",
        description="Data processing and transformation server for multi-server workflows"
    )

    server = ProcessingServer(config)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
